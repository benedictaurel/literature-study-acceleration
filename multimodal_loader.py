
import os
import re
import fitz  # pymupdf
from langchain_core.documents import Document
from typing import List, Dict

# Mapping of common Computer Modern font glyph names to LaTeX/Unicode equivalents
MATH_GLYPH_MAP = {
    # Greek letters
    "alpha": "α", "beta": "β", "gamma": "γ", "delta": "δ", "epsilon": "ε",
    "zeta": "ζ", "eta": "η", "theta": "θ", "iota": "ι", "kappa": "κ",
    "lambda": "λ", "mu": "μ", "nu": "ν", "xi": "ξ", "omicron": "ο",
    "pi": "π", "rho": "ρ", "sigma": "σ", "tau": "τ", "upsilon": "υ",
    "phi": "φ", "chi": "χ", "psi": "ψ", "omega": "ω",
    "Alpha": "Α", "Beta": "Β", "Gamma": "Γ", "Delta": "Δ", "Epsilon": "Ε",
    "Zeta": "Ζ", "Eta": "Η", "Theta": "Θ", "Iota": "Ι", "Kappa": "Κ",
    "Lambda": "Λ", "Mu": "Μ", "Nu": "Ν", "Xi": "Ξ", "Omicron": "Ο",
    "Pi": "Π", "Rho": "Ρ", "Sigma": "Σ", "Tau": "Τ", "Upsilon": "Υ",
    "Phi": "Φ", "Chi": "Χ", "Psi": "Ψ", "Omega": "Ω",
    
    # Math operators and symbols (CMEX10, CMSY10, etc.)
    "parenleftbigg": "(", "parenrightbigg": ")",
    "parenleftBigg": "(", "parenrightBigg": ")",
    "parenleftbig": "(", "parenrightbig": ")",
    "bracketleftbigg": "[", "bracketrightbigg": "]",
    "bracketleftBigg": "[", "bracketrightBigg": "]",
    "braceleftbigg": "{", "bracerightbigg": "}",
    "braceleftBigg": "{", "bracerightBigg": "}",
    "barbigg": "|", "barBigg": "|",
    "summation": "∑", "summationdisplay": "∑",
    "product": "∏", "productdisplay": "∏",
    "integral": "∫", "integraldisplay": "∫",
    "contintegral": "∮", "contintegraldisplay": "∮",
    "radical": "√", "radicalBig": "√", "radicalbigg": "√",
    "infinity": "∞", "infty": "∞",
    "partial": "∂", "nabla": "∇",
    "plusminus": "±", "minusplus": "∓",
    "times": "×", "divide": "÷",
    "leq": "≤", "geq": "≥", "neq": "≠",
    "approx": "≈", "equiv": "≡", "sim": "∼",
    "subset": "⊂", "superset": "⊃", "subseteq": "⊆", "supseteq": "⊇",
    "in": "∈", "notin": "∉", "ni": "∋",
    "forall": "∀", "exists": "∃", "nexists": "∄",
    "emptyset": "∅", "varnothing": "∅",
    "cup": "∪", "cap": "∩",
    "vee": "∨", "wedge": "∧", "neg": "¬",
    "rightarrow": "→", "leftarrow": "←", "leftrightarrow": "↔",
    "Rightarrow": "⇒", "Leftarrow": "⇐", "Leftrightarrow": "⇔",
    "mapsto": "↦", "hookrightarrow": "↪", "hookleftarrow": "↩",
    "uparrow": "↑", "downarrow": "↓", "updownarrow": "↕",
    "cdot": "·", "cdots": "⋯", "ldots": "…", "vdots": "⋮", "ddots": "⋱",
    "prime": "′", "doubleprime": "″",
    "degree": "°",
    "bullet": "•", "circ": "∘",
    "star": "⋆", "ast": "∗",
    "dagger": "†", "ddagger": "‡",
    "aleph": "ℵ", "beth": "ℶ",
    "hbar": "ℏ", "ell": "ℓ",
    "wp": "℘", "Re": "ℜ", "Im": "ℑ",
    "angle": "∠", "measuredangle": "∡",
    "triangle": "△", "square": "□", "diamond": "◇",
    
    # CMR10 special characters
    "suppress": "", # This is typically a zero-width character, can be ignored
    "fi": "fi", "fl": "fl", "ff": "ff", "ffi": "ffi", "ffl": "ffl",
    
    # Superscripts and subscripts
    "zero.superior": "⁰", "one.superior": "¹", "two.superior": "²",
    "three.superior": "³", "four.superior": "⁴", "five.superior": "⁵",
    "six.superior": "⁶", "seven.superior": "⁷", "eight.superior": "⁸",
    "nine.superior": "⁹",
    "zero.inferior": "₀", "one.inferior": "₁", "two.inferior": "₂",
    "three.inferior": "₃", "four.inferior": "₄", "five.inferior": "₅",
    "six.inferior": "₆", "seven.inferior": "₇", "eight.inferior": "₈",
    "nine.inferior": "₉",
}


class MultimodalPDFLoader:
    def __init__(self, file_path: str, image_output_dir: str = "./extracted_images"):
        self.file_path = file_path
        self.image_output_dir = image_output_dir
        if not os.path.exists(self.image_output_dir):
            os.makedirs(self.image_output_dir)

    def load(self) -> List[Document]:
        """
        Loads the PDF and extracts text and images.
        Returns a list of LangChain Documents.
        Mathematical fonts are converted to Unicode equivalents.
        """
        documents = []
        doc = fitz.open(self.file_path)
        file_basename = os.path.basename(self.file_path)

        for page_num in range(len(doc)):
            page = doc[page_num]
            
            # 1. Extract Images
            image_paths = self._extract_images(doc, page, page_num, file_basename)

            # 2. Extract Text with math symbol handling
            text = self._extract_text_with_math(page)
            
            if text.strip():
                metadata = {
                    "source": self.file_path,
                    "page": page_num + 1,
                    "type": "text",
                    "image_paths": ",".join(image_paths)
                }
                documents.append(Document(page_content=text, metadata=metadata))
        
        doc.close()
        return documents
    
    def _extract_text_with_math(self, page) -> str:
        """
        Extract text from a page, converting math font glyphs to Unicode.
        Uses PyMuPDF's dict extraction for detailed font/char info.
        """
        # First try: Get the standard text (often works for many PDFs)
        standard_text = page.get_text("text")
        
        # Check if there might be missing characters (indicated by replacement char or empty spans)
        if self._has_potential_math_issues(standard_text):
            # Use detailed extraction to handle math fonts
            try:
                detailed_text = self._extract_with_font_mapping(page)
                if detailed_text.strip():
                    return detailed_text
            except Exception as e:
                print(f"Warning: Detailed math extraction failed: {e}")
        
        return standard_text
    
    def _has_potential_math_issues(self, text: str) -> bool:
        """
        Check if text might have math font extraction issues.
        """
        # Look for replacement characters or suspicious patterns
        if '\ufffd' in text:  # Unicode replacement character
            return True
        if '�' in text:
            return True
        # Check for patterns that suggest missing math (empty formula placeholders)
        if re.search(r'\(\s*\)', text) or re.search(r'\[\s*\]', text):
            return True
        return False
    
    def _extract_with_font_mapping(self, page) -> str:
        """
        Extract text using detailed block/line/span structure,
        mapping math font glyph names to Unicode.
        """
        blocks = page.get_text("dict")["blocks"]
        result_lines = []
        
        for block in blocks:
            if block["type"] != 0:  # Skip non-text blocks (images)
                continue
            
            for line in block.get("lines", []):
                line_text = ""
                for span in line.get("spans", []):
                    span_text = span.get("text", "")
                    font_name = span.get("font", "")
                    
                    # Check if this is a math font (CM = Computer Modern, etc.)
                    if any(mf in font_name.upper() for mf in ["CMEX", "CMSY", "CMMI", "CMR", "CMTI", "MSBM", "MSAM"]):
                        # Try to get character-level info for better mapping
                        span_text = self._map_math_chars(span_text, font_name)
                    
                    line_text += span_text
                
                if line_text.strip():
                    result_lines.append(line_text)
        
        return "\n".join(result_lines)
    
    def _map_math_chars(self, text: str, font_name: str) -> str:
        """
        Map individual characters from math fonts to readable equivalents.
        """
        result = []
        for char in text:
            # Check if character is a placeholder or unmapped
            if ord(char) < 32 and char not in '\n\r\t':
                # Try to find a mapping based on character code
                # Many CM fonts use specific positions for certain symbols
                mapped = self._get_cm_char_mapping(ord(char), font_name)
                result.append(mapped if mapped else char)
            else:
                result.append(char)
        
        return "".join(result)
    
    def _get_cm_char_mapping(self, char_code: int, font_name: str) -> str:
        """
        Map Computer Modern font character codes to Unicode.
        """
        font_upper = font_name.upper()
        
        # CMEX10 (Computer Modern Extended) mappings
        if "CMEX" in font_upper:
            cmex_map = {
                0: "(", 1: ")",  # parentheses
                2: "[", 3: "]",  # brackets
                4: "⌊", 5: "⌋",  # floor
                6: "⌈", 7: "⌉",  # ceiling
                8: "{", 9: "}",  # braces
                10: "⟨", 11: "⟩",  # angle brackets
                16: "(", 17: ")",  # big parentheses
                18: "(", 19: ")",  # bigg parentheses
                32: "",  # suppress
                80: "∑", 81: "∏",  # sum, product
                82: "∫", 83: "∮",  # integral, contour integral
                112: "√",  # radical
            }
            return cmex_map.get(char_code, "")
        
        # CMSY10 (Computer Modern Symbol) mappings
        if "CMSY" in font_upper:
            cmsy_map = {
                0: "−", 1: "·", 2: "×", 3: "∗",
                4: "÷", 5: "◇", 6: "±", 7: "∓",
                8: "⊕", 9: "⊖", 10: "⊗", 11: "⊘",
                14: "∘", 15: "•",
                16: "≍", 17: "≡", 18: "⊆", 19: "⊇",
                20: "≤", 21: "≥", 22: "≼", 23: "≽",
                24: "∼", 25: "≈", 26: "⊂", 27: "⊃",
                28: "≪", 29: "≫", 30: "≺", 31: "≻",
                32: "←", 33: "→", 34: "↑", 35: "↓",
                36: "↔", 37: "↗", 38: "↘", 39: "≃",
                40: "⇐", 41: "⇒", 42: "⇑", 43: "⇓",
                44: "⇔", 45: "↖", 46: "↙",
                48: "∝", 49: "′", 50: "∞", 51: "∈",
                52: "∋", 53: "△", 54: "▽",
                56: "∀", 57: "∃", 58: "¬",
                60: "⊤", 61: "⊥", 62: "ℵ",
                102: "{", 103: "}",
                106: "|",
            }
            return cmsy_map.get(char_code, "")
        
        # CMMI (Computer Modern Math Italic) - Greek letters
        if "CMMI" in font_upper:
            cmmi_map = {
                11: "α", 12: "β", 13: "γ", 14: "δ", 15: "ε",
                16: "ζ", 17: "η", 18: "θ", 19: "ι", 20: "κ",
                21: "λ", 22: "μ", 23: "ν", 24: "ξ", 25: "π",
                26: "ρ", 27: "σ", 28: "τ", 29: "υ", 30: "φ",
                31: "χ", 32: "ψ", 33: "ω", 34: "ε", 35: "ϑ",
                36: "ϖ", 37: "ϱ", 38: "ς", 39: "ϕ",
                0: "Γ", 1: "Δ", 2: "Θ", 3: "Λ", 4: "Ξ",
                5: "Π", 6: "Σ", 7: "Υ", 8: "Φ", 9: "Ψ", 10: "Ω",
                58: ".", 59: ",", 60: "<", 62: ">",
                61: "/",
            }
            return cmmi_map.get(char_code, "")
        
        # CMR10 (Computer Modern Roman)
        if "CMR" in font_upper:
            cmr_map = {
                11: "ff", 12: "fi", 13: "fl", 14: "ffi", 15: "ffl",
                16: "ı", 17: "ȷ",  # dotless i, j
                18: "`", 19: "´", 20: "ˇ", 21: "˘", 22: "¯",
                23: "˚", 24: "¸", 25: "ß", 26: "æ", 27: "œ",
                28: "ø", 29: "Æ", 30: "Œ", 31: "Ø",
                32: "",  # suppress (zero-width)
            }
            return cmr_map.get(char_code, "")
        
        return ""

    def _extract_images(self, doc, page, page_num, file_basename) -> List[str]:
        image_paths = []
        images = page.get_images(full=True)
        
        for idx, img_info in enumerate(images):
            xref = img_info[0]
            try:
                base_image = doc.extract_image(xref)
                image_bytes = base_image["image"]
                image_ext = base_image["ext"]
                
                image_filename = f"{file_basename}_p{page_num+1}_i{idx}.{image_ext}"
                image_path = os.path.join(self.image_output_dir, image_filename)
                
                # Save image to disk
                with open(image_path, "wb") as f:
                    f.write(image_bytes)
                    
                image_paths.append(os.path.abspath(image_path))
            except Exception as e:
                print(f"Warning: Failed to extract image {idx} from page {page_num+1}: {e}")
            
        return image_paths
