#!/usr/bin/env python3
"""
linkedin_scrape_and_apply.py

- Scrapes LinkedIn job links from search result pages for given keywords + location
- Fetches job descriptions
- Ranks jobs by TF-IDF similarity with your resume
- Semi-automates Easy Apply (fills resume/cover/phone; pauses before final Submit)

USAGE (example):
python linkedin_scrape_and_apply.py \
  --resume "/home/adya/resume.pdf" \
  --location "Bengaluru, India" \
  --keywords "python,ml,computer vision" \
  --collect 60 \
  --top 7 \
  --profile "C:/Users/Adya/AppData/Local/Google/Chrome/User Data" \
  --profile-dir "Default" \
  --cover-file "/home/adya/cover.txt"
"""

import argparse, os, time, random, re, sys
from urllib.parse import quote_plus
from collections import OrderedDict
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager

# ---------------- resume parsing ----------------
# add imports near top
try:
    import docx2txt
except Exception:
    docx2txt = None

# prefer pdfplumber if available, otherwise use PyPDF2
try:
    import pdfplumber
except Exception:
    pdfplumber = None

try:
    from PyPDF2 import PdfReader
except Exception:
    PdfReader = None

def read_pdf_with_fallback(path):
    path = os.path.expanduser(path)
    if pdfplumber:
        text = []
        with pdfplumber.open(path) as pdf:
            for p in pdf.pages:
                text.append(p.extract_text() or "")
        return "\n".join(text)
    elif PdfReader:
        text = []
        with open(path, "rb") as f:
            reader = PdfReader(f)
            for p in reader.pages:
                try:
                    text.append(p.extract_text() or "")
                except Exception:
                    # page-level extraction failed; continue
                    text.append("")
        return "\n".join(text)
    else:
        raise RuntimeError("No PDF parser installed. Install pdfplumber (`pip install pdfplumber`) or PyPDF2 (`pip install PyPDF2`).")

def read_docx_with_fallback(path):
    if docx2txt:
        return docx2txt.process(path)
    else:
        # basic fallback: try to read as text
        try:
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                return f.read()
        except Exception:
            raise RuntimeError("docx2txt not installed. Install with `pip install docx2txt` or provide a PDF.")

def read_resume(path):
    path = os.path.expanduser(path)
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    lower = path.lower()
    if lower.endswith(".pdf"):
        return read_pdf_with_fallback(path)
    elif lower.endswith(".docx") or lower.endswith(".doc"):
        return read_docx_with_fallback(path)
    else:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()


def normalize(s):
    return re.sub(r'\s+', ' ', s.strip())

# ---------------- build linkedin search url ----------------
def build_search_url(keyword, location=None, start=0):
    q = quote_plus(keyword)
    url = f"https://www.linkedin.com/jobs/search/?keywords={q}"
    if location:
        url += f"&location={quote_plus(location)}"
    if start and start>0:
        url += f"&start={start}"
    return url

# ---------------- selenium helpers ----------------

def create_driver(profile_dir=None, profile_name="Default", headless=False):
    chrome_opts = Options()
    if profile_dir:
        chrome_opts.add_argument(f"--user-data-dir={profile_dir}")
        chrome_opts.add_argument(f"--profile-directory={profile_name}")
    chrome_opts.add_argument("--start-maximized")
    if headless:
        chrome_opts.add_argument("--headless=new")
        chrome_opts.add_argument("--window-size=1920,1080")

    # Use Service(...) and pass it via service=, not as a positional argument
    service = Service(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service, options=chrome_opts)
    return driver


def scroll_container_collect_links(driver, max_links=50, wait_seconds=1.0):
    """Given a search results page loaded, scroll and collect job links (unique)"""
    links = OrderedDict()
    last_height = driver.execute_script("return document.body.scrollHeight")
    tries = 0
    while len(links) < max_links and tries < 20:
        # find anchor tags that point to jobs view
        anchors = driver.find_elements(By.XPATH, "//a[contains(@href,'/jobs/view/')]")
        for a in anchors:
            href = a.get_attribute("href")
            if href and "/jobs/view/" in href:
                # clean query params - keep unique id part
                links[href.split("?")[0]] = True
                if len(links) >= max_links:
                    break
        # scroll down to load more results
        driver.execute_script("window.scrollBy(0, document.body.scrollHeight*0.7);")
        time.sleep(wait_seconds + random.random()*1.2)
        new_height = driver.execute_script("return document.body.scrollHeight")
        if new_height == last_height:
            tries += 1
        else:
            last_height = new_height
            tries = 0
    return list(links.keys())

def fetch_job_description(driver, job_url, timeout=6):
    """Open job_url in the same tab and extract job title, company, description text (best effort)"""
    driver.get(job_url)
    # small wait for rendering
    try:
        WebDriverWait(driver, timeout).until(EC.presence_of_element_located((By.TAG_NAME, "body")))
    except Exception:
        pass
    time.sleep(1 + random.random()*0.7)
    # heuristics for title/company/desc
    title = ""
    company = ""
    desc = ""
    # title
    try:
        el = driver.find_element(By.XPATH, "//h1")
        title = el.text.strip()
    except Exception:
        pass
    # company
    try:
        el = driver.find_element(By.XPATH, "//a[contains(@href,'/company/') or contains(@class,'topcard__org-name')]")
        company = el.text.strip()
    except Exception:
        # fallback: small span near title
        try:
            el = driver.find_element(By.XPATH, "//span[contains(@class,'topcard__flavor')]")
            company = el.text.strip()
        except Exception:
            pass
    # description: try common containers
    desc_selectors = [
        "//div[contains(@class,'description')]",                 # generic
        "//div[contains(@class,'job-description')]",             # variant
        "//div[contains(@class,'jobs-description')]",           # unified
        "//div[contains(@class,'show-more-less-html__markup')]",# new linkedin container
        "//section[contains(@class,'description')]"             # fallback
    ]
    texts = []
    for sel in desc_selectors:
        try:
            nodes = driver.find_elements(By.XPATH, sel)
            for n in nodes:
                t = n.text.strip()
                if len(t) > 50:
                    texts.append(t)
            if texts:
                break
        except Exception:
            continue
    # Additional fallback: collect many divs and join
    if not texts:
        try:
            big = driver.find_element(By.XPATH, "//div[@id='job-details']")
            if big:
                texts.append(big.text)
        except Exception:
            pass
    desc = "\n".join(texts).strip()
    # final cleanup
    title = normalize(title)
    company = normalize(company)
    desc = normalize(desc)
    return {"url": job_url, "title": title, "company": company, "description": desc}

# ---------------- ranking ----------------
def rank_jobs_by_similarity(resume_text, jobs, top_k=10):
    docs = [resume_text] + [j["description"] or j["title"] + " " + j["company"] for j in jobs]
    # fallback: replace empty descriptions with their title/company to avoid all-empty docs
    for i in range(len(docs)):
        if not docs[i].strip(): docs[i] = " "
    vec = TfidfVectorizer(stop_words="english", max_features=5000)
    X = vec.fit_transform(docs)
    # cosine similarity between resume (row 0) and each job row
    resume_vec = X[0]
    job_vecs = X[1:]
    sims = (job_vecs * resume_vec.T).toarray().reshape(-1)
    ranked_idx = np.argsort(-sims)
    ranked = []
    for idx in ranked_idx[:top_k]:
        j = jobs[idx]
        j_copy = j.copy()
        j_copy["score"] = float(sims[idx])
        ranked.append(j_copy)
    return ranked

# ---------------- Easy Apply (semi-automated) ----------------
def safe_send_keys(el, text):
    try:
        el.clear()
    except Exception:
        pass
    el.send_keys(text)

def click_easy_apply_and_fill(driver, job_url, resume_path=None, cover_text=None, phone=None, auto_submit=False):
    # This re-uses the earlier approach: open URL, click Easy Apply, upload resume, fill textarea, traverse Next buttons, pause before final Submit
    driver.get(job_url); time.sleep(1.2 + random.random()*0.8)
    # scroll a bit
    driver.execute_script("window.scrollTo(0, 500);")
    time.sleep(0.8)
    # find Easy Apply button (several heuristics)
    easy_btn = None
    try:
        easy_btn = WebDriverWait(driver, 4).until(
            EC.element_to_be_clickable((By.XPATH, "//button[.//span[contains(text(),'Easy Apply')]]"))
        )
    except Exception:
        # try different text patterns
        try:
            easy_btn = driver.find_element(By.XPATH, "//button[contains(.,'Apply') and contains(.,'Easy')]")
        except Exception:
            easy_btn = None
    if not easy_btn:
        print(" No Easy Apply found:", job_url)
        return False
    try:
        easy_btn.click()
        print(" Clicked Easy Apply")
    except Exception as e:
        print(" Click failed:", e); return False
    time.sleep(1.2)
    # wait for modal / dialog
    try:
        WebDriverWait(driver, 6).until(EC.presence_of_element_located((By.XPATH, "//div[contains(@role,'dialog')]")))
    except Exception:
        pass
    time.sleep(0.8)
    # Upload resume if file input exists
    if resume_path:
        try:
            file_inputs = driver.find_elements(By.XPATH, "//input[@type='file']")
            for fi in file_inputs:
                if fi.is_displayed():
                    fi.send_keys(os.path.abspath(resume_path))
                    print(" Uploaded resume")
                    time.sleep(1)
                    break
        except Exception as e:
            print(" Resume upload failed:", e)
    # Fill textareas heuristically
    if cover_text:
        try:
            textareas = driver.find_elements(By.TAG_NAME, "textarea")
            filled = False
            for ta in textareas:
                ph = (ta.get_attribute("placeholder") or "").lower()
                name = (ta.get_attribute("name") or "").lower()
                if any(k in ph or k in name for k in ["cover","message","note","why","summary","additional","about"]):
                    safe_send_keys(ta, cover_text)
                    filled = True
                    print(" Filled textarea for cover")
                    break
            if not filled and textareas:
                for ta in textareas:
                    if ta.is_displayed():
                        safe_send_keys(ta, cover_text)
                        print(" Filled first visible textarea")
                        filled = True
                        break
        except Exception as e:
            print(" Cover textarea fill failed:", e)
        # try contenteditable
        if not cover_text.strip() == "":
            try:
                divs = driver.find_elements(By.XPATH, "//div[@contenteditable='true']")
                for d in divs:
                    aria = (d.get_attribute("aria-label") or "").lower()
                    if any(k in aria for k in ["cover","message","summary","why"]):
                        d.click(); d.send_keys(cover_text[:1000])
                        print(" Filled contenteditable cover area")
                        break
            except Exception:
                pass
    # Fill phone if present
    if phone:
        try:
            phone_inputs = driver.find_elements(By.XPATH, "//input[@type='tel' or contains(@name,'phone') or contains(@id,'phone')]")
            for ip in phone_inputs:
                if ip.is_displayed():
                    safe_send_keys(ip, phone)
                    print(" Filled phone")
                    break
        except Exception:
            pass

    # Click Next/Continue until final
    max_steps = 8
    for _ in range(max_steps):
        time.sleep(0.9 + random.random()*0.6)
        # try to find a final submit/review button first
        try:
            submit_btn = driver.find_element(By.XPATH, "//button[.//span[contains(text(),'Submit') or contains(text(),'Apply') or contains(text(),'Done')]]")
            txt = (submit_btn.text or "").lower()
            if any(k in txt for k in ["submit", "apply", "done"]):
                print(" Reached final step (button text):", txt)
                if auto_submit:
                    try:
                        submit_btn.click()
                        print(" Auto-submitted")
                        time.sleep(1)
                        return True
                    except Exception as e:
                        print(" Auto-submit failed:", e)
                        return False
                else:
                    print(" Pausing for manual review. Please submit manually in the browser.")
                    # pause to allow manual submission
                    time.sleep(10 + random.random()*6)
                    return True
        except Exception:
            pass
        # else click Next or Continue if present
        clicked = False
        for xp in ["//button[.//span[contains(text(),'Next')]]", "//button[.//span[contains(text(),'Continue')]]"]:
            try:
                btn = driver.find_element(By.XPATH, xp)
                if btn.is_displayed() and btn.is_enabled():
                    btn.click(); clicked = True; time.sleep(1.0 + random.random()*0.8); break
            except Exception:
                continue
        if not clicked:
            break
    print(" Did not find final submit; exiting modal and continuing.")
    # try to close modal
    try:
        close_btn = driver.find_element(By.XPATH, "//button[contains(@aria-label,'Dismiss') or contains(@aria-label,'Close')]")
        close_btn.click()
    except Exception:
        pass
    return False

# ---------------- full pipeline ----------------
def pipeline(args):
    resume_text = read_resume(args.resume)
    resume_text = normalize(resume_text)
    driver = create_driver(profile_dir=args.profile, profile_name=args.profile_dir, headless=args.headless)
    try:
        all_job_urls = []
        keywords = [k.strip() for k in args.keywords.split(",") if k.strip()]
        for kw in keywords:
            url = build_search_url(kw, location=args.location)
            driver.get(url)
            time.sleep(2 + random.random())
            # try to click filters (Optional: filter for Easy Apply - not always present)
            try:
                # click 'All Filters' -> Easy Apply? (UI changes; skip reliably)
                pass
            except Exception:
                pass
            collected = scroll_container_collect_links(driver, max_links=args.collect)
            print(f"Collected {len(collected)} from keyword '{kw}'")
            all_job_urls.extend(collected)
            # polite pause
            time.sleep(1 + random.random()*1.5)
        # dedupe preserve order
        seen = set(); uniq_urls = []
        for u in all_job_urls:
            if u not in seen:
                seen.add(u); uniq_urls.append(u)
        print("Total unique jobs collected:", len(uniq_urls))
        # fetch job descriptions (limit)
        jobs = []
        for u in uniq_urls[: min(len(uniq_urls), args.collect)]:
            try:
                jd = fetch_job_description(driver, u)
                jobs.append(jd)
                print("Fetched:", jd["title"][:60], "—", jd["company"][:40], "score candidates", len(jd["description"]))
            except Exception as e:
                print("Fetch failed for", u, e)
            time.sleep(0.7 + random.random()*0.8)
        # rank
        ranked = rank_jobs_by_similarity(resume_text, jobs, top_k=args.top)
        print("\nTop ranked jobs:")
        for i,j in enumerate(ranked):
            print(i+1, f"score={j['score']:.4f}", j['title'][:60], "-", j['company'][:40], "\n ", j['url'])
        # confirm and apply (semi-automated)
        to_apply = [j['url'] for j in ranked]
        # optional: write to file
        with open("selected_urls.txt","w") as f:
            for u in to_apply:
                f.write(u+"\n")
        print("\nSaved selected job URLs to selected_urls.txt")
        # Apply loop
        if args.do_apply:
            for u in to_apply:
                print("\n=== Processing:", u)
                ok = click_easy_apply_and_fill(driver, u, resume_path=args.resume, cover_text=(open(args.cover_file).read() if args.cover_file else args.cover_text), phone=args.phone, auto_submit=args.auto_submit)
                # small random delay between jobs
                time.sleep(3 + random.random()*3)
    finally:
        print("Pipeline finished. Closing browser in 3s.")
        time.sleep(3)
        driver.quit()

# ---------------- CLI ----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--resume", required=True)
    ap.add_argument("--location", default="")
    ap.add_argument("--keywords", required=True, help="comma separated keywords")
    ap.add_argument("--collect", type=int, default=50, help="max job links to collect per keyword")
    ap.add_argument("--top", type=int, default=8, help="how many top matched jobs to attempt")
    ap.add_argument("--profile", default=None, help="chrome user-data-dir (so you're logged in)")
    ap.add_argument("--profile-dir", default="Default", help="chrome profile directory name")
    ap.add_argument("--headless", action="store_true")
    ap.add_argument("--do-apply", action="store_true", help="actually run the Easy Apply fills (pauses for manual submit)")
    ap.add_argument("--auto-submit", action="store_true", help="AUTO SUBMIT at final step (risky!)")
    ap.add_argument("--cover-file", default="", help="path to cover letter file to paste into application")
    ap.add_argument("--cover-text", default="", help="cover text inline (used if cover-file not provided)")
    ap.add_argument("--phone", default="", help="phone number to fill if present")
    args = ap.parse_args()
    pipeline(args)

if __name__ == "__main__":
    main()
