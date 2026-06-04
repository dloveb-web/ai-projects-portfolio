import os
import io
import json
import subprocess
import tempfile

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

try:
    import fitz
except ImportError:
    fitz = None

try:
    from docx import Document as DocxDocument
except ImportError:
    DocxDocument = None

try:
    from PIL import Image
except ImportError:
    Image = None

try:
    from transformers import LayoutLMv3Processor, LayoutLMv3ForTokenClassification
    import torch
except ImportError:
    LayoutLMv3Processor = None
    LayoutLMv3ForTokenClassification = None
    torch = None

try:
    import pytesseract
    PYTESSERACT_AVAILABLE = True
except ImportError:
    pytesseract = None
    PYTESSERACT_AVAILABLE = False


def is_tesseract_available():
    try:
        result = subprocess.run(["tesseract", "--version"], capture_output=True, text=True, timeout=5)
        return result.returncode == 0
    except:
        return False


class OCRProcessor:
    def __init__(self):
        self.available = is_tesseract_available()
        if not self.available:
            print("⚠️ Tesseract OCR 不可用，图片OCR功能将受限")
    
    def extract_text_from_image(self, image_bytes, lang="chi_sim+eng"):
        if not self.available:
            return ""
        
        try:
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                tmp.write(image_bytes)
                tmp_path = tmp.name
            
            result = subprocess.run(
                ["tesseract", tmp_path, "stdout", "-l", lang],
                capture_output=True,
                text=True,
                timeout=60
            )
            
            os.unlink(tmp_path)
            
            if result.returncode == 0:
                return result.stdout.strip()
            return ""
        except Exception as e:
            print(f"OCR处理失败：{e}")
            return ""
    
    def extract_text_from_pdf_page(self, page, dpi=200, lang="chi_sim+eng"):
        if not self.available:
            return ""
        
        try:
            mat = fitz.Matrix(dpi / 72, dpi / 72)
            pix = page.get_pixmap(matrix=mat)
            image_bytes = pix.tobytes("png")
            return self.extract_text_from_image(image_bytes, lang)
        except Exception as e:
            print(f"PDF页面OCR转换失败：{e}")
            return ""


class DocumentParser:
    def __init__(self, init_layoutlm=False):
        self.layoutlm_processor = None
        self.layoutlm_model = None
        self.layoutlm_initialized = False
        self.ocr = OCRProcessor()
        if init_layoutlm:
            self._init_layoutlm()
        print(f"📖 文档解析器初始化完成 (OCR: {'可用' if self.ocr.available else '不可用'}, LayoutLM: {'已初始化' if self.layoutlm_initialized else '未加载'})")
    
    def _init_layoutlm(self):
        if self.layoutlm_initialized:
            return
        if not LayoutLMv3Processor or not LayoutLMv3ForTokenClassification or not torch:
            print("LayoutLM依赖未安装，将使用基础解析")
            return
        
        if not PYTESSERACT_AVAILABLE:
            print("⚠️ LayoutLM需要pytesseract库，已跳过LayoutLM初始化")
            print("  如需启用LayoutLM，请执行：pip install pytesseract")
            return
            
        try:
            import signal
            def timeout_handler(signum, frame):
                raise TimeoutError("LayoutLM加载超时")
            
            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(600)
            
            print("⏳ 正在加载LayoutLM模型（首次下载约479MB，请耐心等待...）...")
            self.layoutlm_processor = LayoutLMv3Processor.from_pretrained("microsoft/layoutlmv3-base")
            self.layoutlm_model = LayoutLMv3ForTokenClassification.from_pretrained("microsoft/layoutlmv3-base")
            self.layoutlm_model.eval()
            self.layoutlm_initialized = True
            print("✅ LayoutLM模型加载成功")
            
            signal.alarm(0)
        except Exception as e:
            print(f"LayoutLM初始化失败，将使用基础解析：{e}")
            signal.alarm(0)
    
    def parse_pdf(self, file_path):
        if not fitz:
            raise ImportError("PDF解析需要安装 pymupdf：pip install pymupdf")
            
        doc = fitz.open(file_path)
        pages = []
        
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            native_text = page.get_text()
            page_data = {
                "page_number": page_num + 1,
                "text": native_text,
                "images": [],
                "tables": [],
                "layout_elements": []
            }
            
            if len(native_text.strip()) < 50 and self.ocr.available:
                ocr_text = self.ocr.extract_text_from_pdf_page(page)
                if ocr_text:
                    page_data["text"] = ocr_text
                    page_data["ocr_applied"] = True
            
            if Image:
                try:
                    image_list = page.get_images(full=True)
                    for img_index, img in enumerate(image_list):
                        xref = img[0]
                        base_image = doc.extract_image(xref)
                        image_bytes = base_image["image"]
                        image = Image.open(io.BytesIO(image_bytes))
                        image_data = {
                            "index": img_index,
                            "width": image.width,
                            "height": image.height,
                            "bytes": image_bytes
                        }
                        if self.ocr.available and image.width > 100 and image.height > 100:
                            image_text = self.ocr.extract_text_from_image(image_bytes)
                            if image_text:
                                image_data["extracted_text"] = image_text
                        page_data["images"].append(image_data)
                except Exception as e:
                    pass
            
            try:
                page_data["tables"] = self._extract_tables(page)
            except:
                page_data["tables"] = []
            
            if self.layoutlm_initialized and Image:
                try:
                    mat = fitz.Matrix(2, 2)
                    pix = page.get_pixmap(matrix=mat)
                    page_image_bytes = pix.tobytes("png")
                    layout_elements = self.detect_elements(page_image_bytes)
                    if layout_elements:
                        page_data["layout_elements"] = layout_elements
                except Exception as e:
                    pass
            
            pages.append(page_data)
        
        return {"type": "pdf", "pages": pages}
    
    def _extract_tables(self, page):
        tables = []
        try:
            for block in page.get_text("dict")["blocks"]:
                if block["type"] == 1:
                    tables.append({
                        "bbox": block["bbox"],
                        "lines": block.get("lines", [])
                    })
        except:
            pass
        return tables
    
    def parse_docx(self, file_path):
        if not DocxDocument:
            raise ImportError("DOCX解析需要安装 python-docx：pip install python-docx")
            
        doc = DocxDocument(file_path)
        pages = []
        current_page = {
            "page_number": 1,
            "text": "",
            "images": [],
            "tables": []
        }
        
        for element in doc.element.body:
            if element.tag.endswith("p"):
                current_page["text"] += element.text + "\n" if element.text else "\n"
            elif element.tag.endswith("tbl"):
                current_page["tables"].append(self._parse_docx_table(element))
        
        if Image:
            try:
                for rel in doc.part.rels.values():
                    if "image" in rel.target_ref:
                        image_bytes = rel.target_part.blob
                        image = Image.open(io.BytesIO(image_bytes))
                        current_page["images"].append({
                            "width": image.width,
                            "height": image.height,
                            "bytes": image_bytes
                        })
            except Exception as e:
                print(f"提取DOCX图片失败：{e}")
        
        pages.append(current_page)
        return {"type": "docx", "pages": pages}
    
    def _parse_docx_table(self, table_element):
        rows = []
        for row in table_element.findall(".//{http://schemas.openxmlformats.org/wordprocessingml/2006/main}tr"):
            cells = []
            for cell in row.findall(".//{http://schemas.openxmlformats.org/wordprocessingml/2006/main}tc"):
                text = ""
                for t in cell.findall(".//{http://schemas.openxmlformats.org/wordprocessingml/2006/main}t"):
                    text += t.text if t.text else ""
                cells.append(text)
            rows.append(cells)
        return {"rows": rows}
    
    def parse_image(self, file_path):
        if not Image:
            raise ImportError("图片解析需要安装 pillow：pip install pillow")
            
        image = Image.open(file_path)
        image_bytes = self._image_to_bytes(image)
        
        result = {
            "type": "image",
            "width": image.width,
            "height": image.height,
            "bytes": image_bytes
        }
        
        if self.ocr.available and image.width > 100 and image.height > 100:
            ocr_text = self.ocr.extract_text_from_image(image_bytes)
            if ocr_text:
                result["extracted_text"] = ocr_text
                result["pages"] = [{
                    "page_number": 1,
                    "text": ocr_text,
                    "images": [],
                    "tables": []
                }]
        
        return result
    
    def _image_to_bytes(self, image):
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        return buffer.getvalue()
    
    def detect_elements(self, image_bytes):
        if not self.layoutlm_model or not Image:
            return []
        
        if not PYTESSERACT_AVAILABLE:
            return []
        
        try:
            image = Image.open(io.BytesIO(image_bytes))
            encoding = self.layoutlm_processor(image, return_tensors="pt")
            
            with torch.no_grad():
                outputs = self.layoutlm_model(**encoding)
            
            predictions = outputs.logits.argmax(-1).squeeze().tolist()
            labels = self.layoutlm_model.config.id2label
            
            elements = []
            for token, label_id in zip(encoding["input_ids"][0], predictions):
                label = labels.get(label_id, "O")
                if label != "O":
                    elements.append({
                        "token": self.layoutlm_processor.decode([token]),
                        "label": label
                    })
            
            return elements
        except Exception as e:
            print(f"LayoutLM检测失败：{e}")
            return []
    
    def parse(self, file_path):
        ext = os.path.splitext(file_path)[1].lower()
        
        if ext == ".pdf":
            return self.parse_pdf(file_path)
        elif ext in [".docx", ".doc"]:
            return self.parse_docx(file_path)
        elif ext in [".png", ".jpg", ".jpeg", ".gif", ".bmp"]:
            return self.parse_image(file_path)
        elif ext in [".txt", ".text"]:
            # 支持简单文本文件
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
            return {
                "type": "txt",
                "pages": [
                    {
                        "page_number": 1,
                        "text": content,
                        "images": [],
                        "tables": []
                    }
                ]
            }
        else:
            raise ValueError(f"不支持的文件格式：{ext}")
