import os
import time
import glob
import zipfile
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

class ORANScraper:
    def __init__(self, download_dir="./papers"):
        self.download_dir = os.path.abspath(download_dir)
        if not os.path.exists(self.download_dir):
            os.makedirs(self.download_dir)
        
        options = Options()
        try:
            # Check if running in a restricted env? Default to headless.
            options.add_argument("--headless=new")
        except:
            pass
        options.add_argument("--no-sandbox")
        options.add_argument("--disable-dev-shm-usage")
        options.add_argument("--window-size=1920,1080")
        
        prefs = {
            "download.default_directory": self.download_dir,
            "download.prompt_for_download": False,
            "download.directory_upgrade": True,
            "plugins.always_open_pdf_externally": True 
        }
        options.add_experimental_option("prefs", prefs)
        options.add_argument("--log-level=3")
        
        print("Initializing Chrome Driver...")
        self.driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)

    def search_and_download(self, query):
        """
        Searches for specifications matching the query and downloads them.
        Returns a list of downloaded file paths.
        """
        print(f"Navigating to O-RAN specifications page...")
        try:
            self.driver.get("https://specifications.o-ran.org/specifications")
            
            # 1. Wait for page load (Table presence)
            try:
                WebDriverWait(self.driver, 20).until(
                    EC.presence_of_element_located((By.TAG_NAME, "table"))
                )
                time.sleep(3) 
            except Exception as e:
                print(f"Error waiting for page load: {e}")
                return []

            # 2. Try to find a search bar by Label
            print(f"Searching for '{query}'...")
            search_input = None
            try:
                # Find label with "Title/Designator" and get the input it points to
                label = self.driver.find_element(By.XPATH, "//label[contains(text(), 'Title/Designator')]")
                input_id = label.get_attribute("for")
                if input_id:
                    search_input = self.driver.find_element(By.ID, input_id)
            except:
                pass

            target_links = []
            
            if search_input and search_input.is_displayed():
                print("Found search bar. Typing query...")
                search_input.clear()
                search_input.send_keys(query)
                search_input.send_keys(Keys.RETURN)
                
                # Wait for filter results
                print("Waiting for results to update...")
                time.sleep(5) 
                
                # Get visible download buttons
                print("Scanning visible results...")
                all_dl_btns = self.driver.find_elements(By.XPATH, "//a[contains(text(), 'Download')]")
                print(f"DEBUG: Found {len(all_dl_btns)} 'Download' buttons in DOM. Checking visibility...")
                for btn in all_dl_btns:
                    if btn.is_displayed():
                        target_links.append(btn)
                print(f"DEBUG: Found {len(target_links)} visible download buttons.")
                        
            else:
                print("No search bar found or visible. Scanning page rows directly (Title match only).")
                # Fallback: Scan text in rows
                rows = self.driver.find_elements(By.XPATH, "//table/tbody/tr")
                for row in rows:
                    if query.lower() in row.text.lower():
                        try:
                            btn = row.find_element(By.XPATH, ".//a[contains(text(), 'Download')]")
                            target_links.append(btn)
                        except:
                            pass

            print(f"Found {len(target_links)} relevant documents.")
            
            # 4. Download
            count = 0
            # Limit to 3 files 
            for link in target_links[:3]:
                try:
                    href = link.get_attribute("href")
                    # We can't easily check 'already exists' without clicking, unless we track hrefs.
                    # But hrefs are generic 'download?id=...'
                    # So we just download.
                    
                    print(f"Initiating download...")
                    self.driver.execute_script("arguments[0].scrollIntoView(true);", link)
                    time.sleep(1)
                    link.click()
                    
                    if self._wait_for_new_file(self.download_dir):
                        print("Download detected.")
                        count += 1
                    else:
                        print("Download timed out or no new file.")
                        
                except Exception as e:
                    print(f"Error downloading link: {e}")
                    
        except Exception as e:
            print(f"Scraping failed: {e}")

        # Extract zips
        self._extract_zips_if_any()
        
        # Return all PDFs
        final_files = glob.glob(os.path.join(self.download_dir, "*.pdf"))
        return final_files

    def _wait_for_new_file(self, dir_path, timeout=30):
        initial_files = set(os.listdir(dir_path))
        start = time.time()
        while time.time() - start < timeout:
            current_files = set(os.listdir(dir_path))
            new_files = current_files - initial_files
            if any(f.endswith('.crdownload') or f.endswith('.tmp') for f in new_files):
                time.sleep(1)
                continue
            if new_files:
                return True
            time.sleep(1)
        return False
        
    def _extract_zips_if_any(self):
        zips = glob.glob(os.path.join(self.download_dir, "*.zip"))
        for val in zips:
            try:
                print(f"Extracting {val}...")
                with zipfile.ZipFile(val, 'r') as zip_ref:
                    zip_ref.extractall(self.download_dir)
            except Exception as e:
                print(f"Error extracting zip: {e}")

    def close(self):
        try:
            self.driver.quit()
        except:
            pass

if __name__ == "__main__":
    scraper = ORANScraper()
    try:
        # Test query
        print("Testing with query 'Architecture'...")
        docs = scraper.search_and_download("Architecture")
        print("Downloaded:", docs)
    finally:
        scraper.close()
