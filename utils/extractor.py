#pdf extraction using pdfplumber
import pdfplumber
from typing import List
import re

def extract_text_from_pdf(file_path: str) -> List[str]:
    extracted_pages = []

   
    with pdfplumber.open(file_path) as pdf:
        for page_number, page in enumerate(pdf.pages, start=1):
            #extracting texting, if empty returning none
            #text = page.extract_text() or ""
            text = page.extract_text(layout=True, use_text_flow=True) or ""
            #added layout=True attempts to mimic the visual layout of the page more closely, earlier i was creating single list of strings, ->not efficient for semantic searches
            if not text:
                continue

            #Smart logic, instead of aggressive regex splitting------>
            # Split into lines to analyze structure line-by-line
            lines = text.split('\n')
            structured_text = ""
            
            for i, line in enumerate(lines):
                line = line.strip()
                if not line:
                    continue # Skip empty lines

                #TITLE
                if len(line.split()) < 10 and not line[-1] in '.?!:;':
                    structured_text += f"\n\n## {line}\n"

                #BREAKS
                elif re.match(r'^[\-•*]|\d+\.', line):
                    structured_text += f"\n{line}"

                #BREAKS PT 2
                # C. Detect Paragraph Breaks (Heuristic: Ends with punctuation)
                # Action: Keep the newline (end of thought)
                elif line[-1] in '.?!':
                    structured_text += f"{line}\n"
                
                #ELSE jjust join with the space of the stream.
                else:
                    structured_text += f"{line} "

            #removing whitespace-breaks hard new lines, this combines multiples spaces into one single space and then removes the whitespace accordingly
            #removing hyphens from the word, respon-sibility, making it responsibility.
            #regex finds the hyphen and new line and edits it.
            text = re.sub(r'(\w+)-\s+(\w+)', r'\1\2', text)
            text = re.sub(r'\s+', ' ', text).strip()
            
            extracted_pages.append(text)

            print(f"Extracted Page {page_number}: {len(text)} characters")

    print(f"\nTotal Pages Extracted: {len(extracted_pages)}")
    return extracted_pages
