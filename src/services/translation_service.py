from typing import Protocol, Tuple
from abc import ABC, abstractmethod
import torch
import time
import functools
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from loguru import logger

from src.config.settings import settings



def timer_decorator(func):
    """Decorator to measure execution time of async functions."""
    @functools.wraps(func)
    async def wrapper(self, *args, **kwargs):
        start_time = time.perf_counter()
        try:
            result = await func(self, *args, **kwargs)
            end_time = time.perf_counter()
            execution_time = end_time - start_time
            
            # Log the timing
            logger.info(f"{func.__name__} completed in {execution_time:.3f} seconds")
            
            # Store timing in instance for later retrieval
            self._last_execution_time = execution_time
            
            return result
        except Exception as e:
            end_time = time.perf_counter()
            execution_time = end_time - start_time
            logger.error(f"{func.__name__} failed after {execution_time:.3f} seconds: {e}")
            raise
    return wrapper


class TranslationProvider(Protocol):
    """Protocol for translation providers."""
    
    async def translate(
        self, 
        text: str, 
        source_lang: str, 
        target_lang: str
    ) -> Tuple[str, float]:
        """Translate text from source to target language."""
        ...


class NLLBTranslationService:
    """NLLB-200 translation service implementation."""
    
    def __init__(self) -> None:
        self.model = None
        self.tokenizer = None
        self._is_loaded = False
        self._last_execution_time = 0.0
        self._language_mapping = {
            'achinese': 'ace_Arab',
            'mesopotamian arabic': 'acm_Arab',
            'taizzi-adeni arabic': 'acq_Arab',
            'tunisian arabic': 'aeb_Arab',
            'afrikaans': 'afr_Latn',
            'ajp': 'ajp_Arab',
            'akan': 'aka_Latn',
            'tosk albanian': 'als_Latn',
            'amharic': 'amh_Ethi',
            'levantine arabic': 'apc_Arab',
            'arabic': 'arb_Arab',
            'najdi arabic': 'ars_Arab',
            'moroccan arabic': 'ary_Arab',
            'egyptian arabic': 'arz_Arab',
            'assamese': 'asm_Beng',
            'asturian': 'ast_Latn',
            'awadhi': 'awa_Deva',
            'central aymara': 'ayr_Latn',
            'south azerbaijani': 'azb_Arab',
            'north azerbaijani': 'azj_Latn',
            'bashkir': 'bak_Cyrl',
            'bambara': 'bam_Latn',
            'balinese': 'ban_Latn',
            'belarusian': 'bel_Cyrl',
            'bemba (zambia)': 'bem_Latn',
            'bengali': 'ben_Beng',
            'bhojpuri': 'bho_Deva',
            'banjar': 'bjn_Latn',
            'tibetan': 'bod_Tibt',
            'bosnian': 'bos_Latn',
            'buginese': 'bug_Latn',
            'bulgarian': 'bul_Cyrl',
            'catalan': 'cat_Latn',
            'cebuano': 'ceb_Latn',
            'czech': 'ces_Latn',
            'chokwe': 'cjk_Latn',
            'central kurdish': 'ckb_Arab',
            'crimean tatar': 'crh_Latn',
            'welsh': 'cym_Latn',
            'danish': 'dan_Latn',
            'german': 'deu_Latn',
            'southwestern dinka': 'dik_Latn',
            'dyula': 'dyu_Latn',
            'dzongkha': 'dzo_Tibt',
            'greek': 'ell_Grek',
            'english': 'eng_Latn',
            'esperanto': 'epo_Latn',
            'estonian': 'est_Latn',
            'basque': 'eus_Latn',
            'ewe': 'ewe_Latn',
            'faroese': 'fao_Latn',
            'fijian': 'fij_Latn',
            'finnish': 'fin_Latn',
            'fon': 'fon_Latn',
            'french': 'fra_Latn',
            'friulian': 'fur_Latn',
            'nigerian fulfulde': 'fuv_Latn',
            'west central oromo': 'gaz_Latn',
            'scottish gaelic': 'gla_Latn',
            'irish': 'gle_Latn',
            'galician': 'glg_Latn',
            'guaraní': 'grn_Latn',
            'gujarati': 'guj_Gujr',
            'haitian': 'hat_Latn',
            'hausa': 'hau_Latn',
            'hebrew': 'heb_Hebr',
            'hindi': 'hin_Deva',
            'chhattisgarhi': 'hne_Deva',
            'croatian': 'hrv_Latn',
            'hungarian': 'hun_Latn',
            'armenian': 'hye_Armn',
            'igbo': 'ibo_Latn',
            'iloko': 'ilo_Latn',
            'indonesian': 'ind_Latn',
            'icelandic': 'isl_Latn',
            'italian': 'ita_Latn',
            'javanese': 'jav_Latn',
            'japanese': 'jpn_Jpan',
            'kabyle': 'kab_Latn',
            'kachin': 'kac_Latn',
            'kamba (kenya)': 'kam_Latn',
            'kannada': 'kan_Knda',
            'kashmiri': 'kas_Arab',
            'georgian': 'kat_Geor',
            'kazakh': 'kaz_Cyrl',
            'kabiyè': 'kbp_Latn',
            'kabuverdianu': 'kea_Latn',
            'halh mongolian': 'khk_Cyrl',
            'khmer': 'khm_Khmr',
            'kikuyu': 'kik_Latn',
            'kinyarwanda': 'kin_Latn',
            'kyrgyz': 'kir_Cyrl',
            'kimbundu': 'kmb_Latn',
            'northern kurdish': 'kmr_Latn',
            'central kanuri': 'knc_Latn',
            'kongo': 'kon_Latn',
            'korean': 'kor_Hang',
            'lao': 'lao_Laoo',
            'ligurian': 'lij_Latn',
            'limburgish': 'lim_Latn',
            'lingala': 'lin_Latn',
            'lithuanian': 'lit_Latn',
            'lombard': 'lmo_Latn',
            'latgalian': 'ltg_Latn',
            'luxembourgish': 'ltz_Latn',
            'luba-lulua': 'lua_Latn',
            'ganda': 'lug_Latn',
            'luo (kenya and tanzania)': 'luo_Latn',
            'lushai': 'lus_Latn',
            'standard latvian': 'lvs_Latn',
            'magahi': 'mag_Deva',
            'maithili': 'mai_Deva',
            'malayalam': 'mal_Mlym',
            'marathi': 'mar_Deva',
            'minangkabau': 'min_Latn',
            'macedonian': 'mkd_Cyrl',
            'maltese': 'mlt_Latn',
            'manipuri': 'mni_Beng',
            'mossi': 'mos_Latn',
            'māori': 'mri_Latn',
            'burmese': 'mya_Mymr',
            'dutch': 'nld_Latn',
            'norwegian nynorsk': 'nno_Latn',
            'norwegian bokmål': 'nob_Latn',
            'nepali (individual language)': 'npi_Deva',
            'pedi': 'nso_Latn',
            'nuer': 'nus_Latn',
            'chichewa': 'nya_Latn',
            'occitan': 'oci_Latn',
            'odia': 'ory_Orya',
            'pangasinan': 'pag_Latn',
            'panjabi': 'pan_Guru',
            'papiamento': 'pap_Latn',
            'southern pashto': 'pbt_Arab',
            'iranian persian': 'pes_Arab',
            'plateau malagasy': 'plt_Latn',
            'polish': 'pol_Latn',
            'portuguese': 'por_Latn',
            'dari': 'prs_Arab',
            'ayacucho quechua': 'quy_Latn',
            'romanian': 'ron_Latn',
            'kirundi': 'run_Latn',
            'russian': 'rus_Cyrl',
            'sango': 'sag_Latn',
            'sanskrit': 'san_Deva',
            'santali': 'sat_Olck',
            'sicilian': 'scn_Latn',
            'shan': 'shn_Mymr',
            'sinhala': 'sin_Sinh',
            'slovak': 'slk_Latn',
            'slovenian': 'slv_Latn',
            'samoan': 'smo_Latn',
            'shona': 'sna_Latn',
            'sindhi': 'snd_Arab',
            'somali': 'som_Latn',
            'southern sotho': 'sot_Latn',
            'spanish': 'spa_Latn',
            'sardinian': 'srd_Latn',
            'serbian': 'srp_Cyrl',
            'swati': 'ssw_Latn',
            'sundanese': 'sun_Latn',
            'swedish': 'swe_Latn',
            'swahili (individual language)': 'swh_Latn',
            'silesian': 'szl_Latn',
            'tamil': 'tam_Taml',
            'tamasheq': 'taq_Latn',
            'tatar': 'tat_Cyrl',
            'telugu': 'tel_Telu',
            'tajik': 'tgk_Cyrl',
            'tagalog': 'tgl_Latn',
            'thai': 'tha_Thai',
            'tigrinya': 'tir_Ethi',
            'tok pisin': 'tpi_Latn',
            'tswana': 'tsn_Latn',
            'tsonga': 'tso_Latn',
            'turkmen': 'tuk_Latn',
            'tumbuka': 'tum_Latn',
            'turkish': 'tur_Latn',
            'twi': 'twi_Latn',
            'central atlas tamazight': 'tzm_Tfng',
            'uyghur': 'uig_Arab',
            'ukrainian': 'ukr_Cyrl',
            'umbundu': 'umb_Latn',
            'urdu': 'urd_Arab',
            'northern uzbek': 'uzn_Latn',
            'venetian': 'vec_Latn',
            'vietnamese': 'vie_Latn',
            'waray (philippines)': 'war_Latn',
            'wolof': 'wol_Latn',
            'xhosa': 'xho_Latn',
            'eastern yiddish': 'ydd_Hebr',
            'yoruba': 'yor_Latn',
            'yue chinese': 'yue_Hant',
            'chinese': 'zho_Hans',
            'standard malay': 'zsm_Latn',
            'zulu': 'zul_Latn'
        }
        
    async def initialize(self) -> None:
        """Initialize the translation model."""
        try:
            logger.info(f"Loading model: {settings.model_name}")
            
            # Check GPU availability
            if torch.cuda.is_available():
                logger.info(f"CUDA available: {torch.cuda.get_device_name(0)}")
                logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
            else:
                logger.warning("CUDA not available, using CPU")
            
            self.tokenizer = AutoTokenizer.from_pretrained(
                settings.model_name,
                cache_dir=settings.cache_dir
            )
            
            self.model = AutoModelForSeq2SeqLM.from_pretrained(
                settings.model_name,
                cache_dir=settings.cache_dir,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto",
                # Optimize for GPU usage
                low_cpu_mem_usage=True,
                use_cache=True,
                attn_implementation="eager"
            )
            
            # Enable optimizations for inference
            if torch.cuda.is_available():
                self.model = torch.compile(self.model, mode="reduce-overhead")
                # Warm up GPU
                logger.info("Warming up GPU...")
                dummy_input = self.tokenizer("Hello", return_tensors="pt").to(self.model.device)
                with torch.no_grad():
                    self.model.generate(**dummy_input, max_length=10, num_beams=1)
                logger.info("GPU warmed up")
            
            self._is_loaded = True
            logger.info("Model loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    @property
    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        return self._is_loaded
    
    def _get_nllb_lang_code(self, lang: str) -> str:
        """Convert common language names to NLLB language codes."""
        lang_lower = lang.lower()
        if lang_lower in self._language_mapping:
            return self._language_mapping[lang_lower]
        # If already in NLLB format, return as is
        if '_' in lang and len(lang.split('_')) == 2:
            return lang
        raise ValueError(f"Unsupported language: {lang}")
    
    def get_supported_languages(self) -> list[str]:
        """Get list of supported language codes."""
        return list(self._language_mapping.keys())
    
    @property
    def last_execution_time(self) -> float:
        """Get the execution time of the last translation."""
        return self._last_execution_time
    
    @timer_decorator
    async def translate(
        self, 
        text: str, 
        source_lang: str, 
        target_lang: str
    ) -> Tuple[str, float]:
        """Translate text using NLLB-200 model. Returns (translated_text, confidence)."""
        if not self.model or not self.tokenizer:
            raise RuntimeError("Model not initialized")
        
        try:
            # Convert language names to NLLB codes
            source_code = self._get_nllb_lang_code(source_lang)
            target_code = self._get_nllb_lang_code(target_lang)
            
            # Set source language
            self.tokenizer.src_lang = source_code
            
            # Prepare input with optimized settings
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512  # Increased for better quality
            )
            
            # Move to GPU device
            device = next(self.model.parameters()).device
            inputs = {k: v.to(device, non_blocking=True) for k, v in inputs.items()}
            
            # Generate translation with optimized GPU settings
            with torch.no_grad(), torch.amp.autocast('cuda', enabled=torch.cuda.is_available()):
                outputs = self.model.generate(
                    **inputs,
                    forced_bos_token_id=self.tokenizer.convert_tokens_to_ids(target_code),
                    max_length=512,
                    num_beams=1,  # Use greedy decoding for speed
                    do_sample=False,
                    early_stopping=True,
                    pad_token_id=self.tokenizer.pad_token_id,
                    use_cache=True,
                    # GPU optimization settings
                    output_scores=False,
                    return_dict_in_generate=False
                )
            
            # Decode result
            translated_text = self.tokenizer.decode(
                outputs[0],
                skip_special_tokens=True
            )
            
            # Clear GPU cache periodically
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            confidence = 0.8
            
            return translated_text.strip(), confidence

        except Exception as e:
            logger.error(f"Translation failed: {e}")
            raise RuntimeError(f"Translation failed: {str(e)}")


# Global service instance
translation_service = NLLBTranslationService()
