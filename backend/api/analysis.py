"""
Эндпоинты анализа резюме с интеграцией всех анализаторов.

Этот модуль предоставляет эндпоинты для анализа загруженных резюме с использованием
различных ML/NLP анализаторов, включая извлечение ключевых слов, распознавание именованных
сущностей, проверку грамматики и расчёт опыта работы.
"""
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Any

from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

# Добавляем родительскую директорию в path для импорта из сервиса data_extractor
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "services" / "data_extractor"))

from analyzers import (
    extract_resume_keywords_hf as extract_resume_keywords,
    extract_resume_entities,
    check_grammar_resume,
    calculate_total_experience,
    format_experience_summary,
)
from database import get_db
from models.resume import ResumeStatus
from i18n.backend_translations import get_error_message, get_success_message

logger = logging.getLogger(__name__)

router = APIRouter()

# Директория, где хранятся загруженные резюме
UPLOAD_DIR = Path("data/uploads")


def _extract_locale(request: Optional[Request]) -> str:
    """
    Извлечь заголовок Accept-Language из запроса.

    Args:
        request: Входящий запрос FastAPI (опционально)

    Returns:
        Код языка (например, 'en', 'ru')
    """
    if request is None:
        return "en"
    accept_language = request.headers.get("Accept-Language", "en")
    lang_code = accept_language.split("-")[0].split(",")[0].strip().lower()
    return lang_code


class AnalysisRequest(BaseModel):
    """Модель запроса для эндпоинта анализа резюме."""

    resume_id: str = Field(..., description="Уникальный идентификатор резюме для анализа")
    extract_experience: bool = Field(
        default=True, description="Извлекать и рассчитывать информацию об опыте работы"
    )
    check_grammar: bool = Field(
        default=True, description="Выполнять проверку грамматики и орфографии"
    )


class KeywordAnalysis(BaseModel):
    """Результаты извлечения ключевых слов."""

    keywords: List[str] = Field(..., description="Извлеченные ключевые слова")
    keyphrases: List[str] = Field(..., description="Извлеченные ключевые фразы")
    scores: List[float] = Field(..., description="Оценки уверенности для каждого ключевого слова")


class EntityAnalysis(BaseModel):
    """Результаты распознавания именованных сущностей."""

    organizations: List[str] = Field(..., description="Извлеченные названия организаций")
    dates: List[str] = Field(..., description="Извлеченные выражения дат")
    persons: List[str] = Field(default=[], description="Извлеченные имена")
    locations: List[str] = Field(default=[], description="Извлеченные названия мест")
    technical_skills: List[str] = Field(..., description="Извлеченные технические навыки")


class GrammarError(BaseModel):
    """Отдельная грамматическая/орфографическая ошибка."""

    type: str = Field(..., description="Тип ошибки (grammar, spelling, punctuation, style)")
    severity: str = Field(..., description="Уровень серьёзности (error, warning)")
    message: str = Field(..., description="Сообщение об ошибке")
    context: str = Field(..., description="Контекст текста, где возникла ошибка")
    suggestions: List[str] = Field(..., description="Предлагаемые исправления")
    position: Dict[str, int] = Field(..., description="Позиция ошибки в символах")


class GrammarAnalysis(BaseModel):
    """Результаты проверки грамматики."""

    total_errors: int = Field(..., description="Общее количество найденных ошибок")
    errors_by_category: Dict[str, int] = Field(
        ..., description="Разбивка ошибок по типам"
    )
    errors_by_severity: Dict[str, int] = Field(
        ..., description="Разбивка ошибок по серьёзности"
    )
    errors: List[GrammarError] = Field(..., description="Список отдельных ошибок")


class ExperienceEntry(BaseModel):
    """Отдельная запись о работе."""

    company: str = Field(..., description="Название компании")
    position: str = Field(..., description="Должность")
    start_date: str = Field(..., description="Дата начала (в формате ISO)")
    end_date: Optional[str] = Field(..., description="Дата окончания (в формате ISO) или None, если текущее место")
    duration_months: int = Field(..., description="Продолжительность в месяцах")


class ExperienceAnalysis(BaseModel):
    """Результаты расчёта опыта работы."""

    total_months: int = Field(..., description="Общий опыт работы в месяцах")
    total_years: float = Field(..., description="Общий опыт работы в годах")
    total_years_formatted: str = Field(..., description="Человекочитаемая сводка опыта работы")
    entries: List[ExperienceEntry] = Field(..., description="Отдельные записи опыта работы")


class AnalysisResponse(BaseModel):
    """Полный ответ анализа."""

    resume_id: str = Field(..., description="Идентификатор резюме")
    status: str = Field(..., description="Статус анализа")
    language: str = Field(..., description="Обнаруженный язык (en, ru)")
    keywords: KeywordAnalysis = Field(..., description="Результаты извлечения ключевых слов")
    entities: EntityAnalysis = Field(..., description="Результаты распознавания именованных сущностей")
    grammar: Optional[GrammarAnalysis] = Field(
        None, description="Результаты проверки грамматики (если включено)"
    )
    experience: Optional[ExperienceAnalysis] = Field(
        None, description="Результаты расчёта опыта работы (если включено)"
    )
    processing_time_ms: float = Field(..., description="Время обработки анализа в миллисекундах")


def find_resume_file(resume_id: str, locale: str = "en") -> Path:
    """
    Найти файл резюме по идентификатору.

    Args:
        resume_id: Уникальный идентификатор резюме
        locale: Код языка для перевода сообщений об ошибках

    Returns:
        Путь к файлу резюме

    Raises:
        HTTPException: Если файл резюме не найден
    """
    # Пробуем распространённые расширения файлов
    for ext in [".pdf", ".docx", ".PDF", ".DOCX"]:
        file_path = UPLOAD_DIR / f"{resume_id}{ext}"
        if file_path.exists():
            return file_path

    # Если не найден, возвращаем ошибку
    error_msg = get_error_message("file_not_found", locale)
    raise HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail=error_msg,
    )


def extract_text_from_file(file_path: Path, locale: str = "en") -> str:
    """
    Извлечь текст из файла резюме (PDF или DOCX).

    Args:
        file_path: Путь к файлу резюме
        locale: Код языка для перевода сообщений об ошибках

    Returns:
        Извлеченный текстовый контент

    Raises:
        HTTPException: Если извлечение текста не удалось
    """
    try:
        # Импорт функций извлечения
        from services.data_extractor.extract import extract_text_from_pdf, extract_text_from_docx

        file_ext = file_path.suffix.lower()

        if file_ext == ".pdf":
            result = extract_text_from_pdf(file_path)
        elif file_ext == ".docx":
            result = extract_text_from_docx(file_path)
        else:
            error_msg = get_error_message("invalid_file_type", locale, file_ext=file_ext, allowed=".pdf, .docx")
            raise HTTPException(
                status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
                detail=error_msg,
            )

        # Проверка ошибок извлечения
        if result.get("error"):
            error_msg = get_error_message("extraction_failed", locale)
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=error_msg,
            )

        text = result.get("text", "")
        if not text or len(text.strip()) < 10:
            error_msg = get_error_message("file_corrupted", locale)
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=error_msg,
            )

        logger.info(f"Извлечено {len(text)} символов из {file_path.name}")
        return text

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Ошибка извлечения текста из {file_path}: {e}", exc_info=True)
        error_msg = get_error_message("extraction_failed", locale)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=error_msg,
        ) from e


@router.post(
    "/analyze",
    response_model=AnalysisResponse,
    status_code=status.HTTP_200_OK,
    tags=["Analysis"],
)
async def analyze_resume(http_request: Request, request: AnalysisRequest, db: AsyncSession = Depends(get_db)) -> JSONResponse:
    """
    Проанализировать резюме с использованием интегрированных ML/NLP анализаторов.

    Этот эндпоинт выполняет комплексный анализ резюме, включая:
    - Извлечение ключевых слов (KeyBERT)
    - Распознавание именованных сущностей (SpaCy)
    - Проверку грамматики и орфографии (LanguageTool)
    - Расчёт опыта работы

    Результаты сохраняются в базу данных для будущего использования.

    Args:
        http_request: Объект запроса FastAPI (для заголовка Accept-Language)
        request: Запрос на анализ с resume_id и опциями анализа
        db: Сессия базы данных

    Returns:
        JSON-ответ с полными результатами анализа

    Raises:
        HTTPException(404): Если файл резюме не найден
        HTTPException(422): Если извлечение текста не удалось
        HTTPException(500): Если обработка анализа не удалась

    Examples:
        >>> import requests
        >>> response = requests.post(
        ...     "http://localhost:8000/api/resumes/analyze",
        ...     json={"resume_id": "abc123", "check_grammar": True, "extract_experience": True}
        ... )
        >>> response.json()
        {
            "resume_id": "abc123",
            "status": "completed",
            "language": "en",
            "keywords": {...},
            "entities": {...},
            "grammar": {...},
            "experience": {...},
            "processing_time_ms": 1234.56
        }
    """
    import time
    from uuid import UUID
    from analyzers import save_resume_analysis, calculate_quality_score
    from models.resume import Resume as ResumeModel
    from models.resume_analysis import ResumeAnalysis

    # Извлечение локали из заголовка Accept-Language
    locale = _extract_locale(http_request)

    start_time = time.time()

    try:
        logger.info(f"Начало анализа для resume_id: {request.resume_id}")

        # Шаг 1: Найти файл резюме
        file_path = find_resume_file(request.resume_id, locale)
        logger.info(f"Найден файл резюме: {file_path}")

        # Шаг 2: Извлечь текст из файла
        resume_text = extract_text_from_file(file_path, locale)

        # Шаг 3: Определить язык текста
        try:
            from langdetect import detect, LangDetectException

            try:
                detected_lang = detect(resume_text)
                # Нормализация к поддерживаемым языкам
                language = "ru" if detected_lang == "ru" else "en"
            except LangDetectException:
                logger.warning("Определение языка не удалось, используем английский по умолчанию")
                language = "en"
        except ImportError:
            logger.warning("langdetect не установлен, используем английский по умолчанию")
            language = "en"

        logger.info(f"Обнаруженный язык: {language}")

        # Шаг 4: Извлечение ключевых слов
        logger.info("Извлечение ключевых слов...")
        keywords_result = extract_resume_keywords(
            resume_text, language=language
        )

        # Обработка различных форматов возврата от экстракторов
        # HF экстрактор возвращает: single_words, keyphrases, all_keywords
        # Старый формат возвращает: keywords, keyphrases, scores
        # Значения могут быть кортежами (keyword, score) - извлекаем только строку ключевого слова
        def extract_strings(items):
            """Извлечь строковые ключевые слова из кортежей или списков."""
            result = []
            for item in items:
                if isinstance(item, (list, tuple)):
                    # Первый элемент - ключевое слово, второй - оценка
                    result.append(str(item[0]) if item else "")
                else:
                    result.append(str(item))
            return result

        if "single_words" in keywords_result:
            # Формат HF - конвертируем в ожидаемый формат
            single_words = keywords_result.get("single_words", [])
            keywords_list = extract_strings(single_words)
            keyphrases_list = extract_strings(keywords_result.get("keyphrases", []))
            keyword_analysis = KeywordAnalysis(
                keywords=keywords_list,
                keyphrases=keyphrases_list,
                scores=[],  # Оценки недоступны в этом формате
            )
        else:
            # Старый формат - также извлекаем строки
            keywords_data = keywords_result.get("keywords", [])
            keyphrases_data = keywords_result.get("keyphrases", [])
            keyword_analysis = KeywordAnalysis(
                keywords=extract_strings(keywords_data),
                keyphrases=extract_strings(keyphrases_data),
                scores=keywords_result.get("scores", []),
            )

        # Шаг 5: Распознавание именованных сущностей
        logger.info("Распознавание именованных сущностей...")
        entities_result = extract_resume_entities(resume_text, language=language)

        # Обработка обоих имён полей 'skills' и 'technical_skills' от разных экстракторов
        skills = entities_result.get("technical_skills") or entities_result.get("skills") or []

        entity_analysis = EntityAnalysis(
            organizations=entities_result.get("organizations") or [],
            dates=entities_result.get("dates") or [],
            persons=entities_result.get("persons") or [],
            locations=entities_result.get("locations") or [],
            technical_skills=skills,
        )

        # Шаг 6: Проверка грамматики (опционально)
        grammar_analysis = None
        if request.check_grammar:
            logger.info("Проверка грамматики...")
            try:
                grammar_result = check_grammar_resume(resume_text, language=language)

                # Конвертация грамматических ошибок в модели ответов
                error_models = []
                for error in grammar_result.get("errors", []):
                    error_models.append(
                        GrammarError(
                            type=error.get("type", "unknown"),
                            severity=error.get("severity", "warning"),
                            message=error.get("message", ""),
                            context=error.get("context", ""),
                            suggestions=error.get("suggestions", []),
                            position=error.get("position", {}),
                        )
                    )

                grammar_analysis = GrammarAnalysis(
                    total_errors=grammar_result.get("total_errors", 0),
                    errors_by_category=grammar_result.get("errors_by_category", {}),
                    errors_by_severity=grammar_result.get("errors_by_severity", {}),
                    errors=error_models,
                )

                logger.info(
                    f"Найдено {grammar_analysis.total_errors} грамматических/орфографических ошибок"
                )
            except Exception as e:
                logger.warning(f"Проверка грамматики не удалась: {e}")
                # Продолжаем без результатов грамматики, а не прерываем весь анализ

        # Шаг 7: Расчёт опыта работы (опционально, требует структурированных данных)
        experience_analysis = None
        total_experience_months = 0
        if request.extract_experience:
            logger.info("Расчёт опыта работы...")
            try:
                # Расчитываем общий опыт работы в месяцах
                total_experience_months = calculate_total_experience(resume_text)
                experience_analysis = ExperienceAnalysis(
                    total_months=total_experience_months,
                    total_years=round(total_experience_months / 12, 1),
                    total_years_formatted=format_experience_summary(total_experience_months),
                    entries=[],
                )
                logger.info(f"Расчитан опыт работы: {experience_analysis.total_years_formatted}")
            except Exception as e:
                logger.warning(f"Расчёт опыта не удался: {e}", exc_info=True)
                # Продолжаем без результатов опыта

        # Расчёт времени обработки
        processing_time_ms = (time.time() - start_time) * 1000

        # Подготовка списка навыков для сохранения
        skills_list = entity_analysis.technical_skills

        # Подготовка ключевых слов для сохранения
        keywords_list = []
        if keyword_analysis.keywords:
            for i, kw in enumerate(keyword_analysis.keywords):
                score = keyword_analysis.scores[i] if i < len(keyword_analysis.scores) else 0.8
                keywords_list.append({"keyword": kw, "score": score})

        # Расчёт оценки качества
        quality_score = calculate_quality_score(
            grammar_issues=grammar_analysis.errors if grammar_analysis else None,
            has_contact_info=bool(resume_text.find('@') > 0),
            text_length=len(resume_text),
        )

        # Сохранение анализа в базу данных
        try:
            await save_resume_analysis(
                db=db,
                resume_id=UUID(request.resume_id),
                raw_text=resume_text[:50000],  # Сохраняем первые 50к символов
                language=language,
                skills=skills_list,
                keywords=keywords_list,
                entities=entity_analysis.model_dump(),
                quality_score=quality_score,
                processing_time_seconds=processing_time_ms / 1000,
                analyzer_version="2.0.0",
                grammar_issues=[e.model_dump() for e in grammar_analysis.errors] if grammar_analysis else None,
            )

            # Обновление статуса резюме и языка
            resume_update = await db.execute(
                select(ResumeModel).where(ResumeModel.id == UUID(request.resume_id))
            )
            resume_obj = resume_update.scalar_one_or_none()
            if resume_obj:
                resume_obj.status = ResumeStatus.COMPLETED
                resume_obj.language = language
                resume_obj.raw_text = resume_text[:10000]
                # Сохраняем рассчитанный опыт работы
                if total_experience_months > 0:
                    resume_obj.total_experience_months = total_experience_months
                await db.commit()

            logger.info(f"Анализ сохранён в базу данных для resume_id {request.resume_id}")
        except Exception as e:
            logger.error(f"Не удалось сохранить анализ в базу данных: {e}", exc_info=True)
            # Продолжаем в любом случае - анализ прошёл успешно, только сохранение в БД не удалось

        # Формирование ответа
        response_data = {
            "resume_id": request.resume_id,
            "status": "completed",
            "language": language,
            "keywords": keyword_analysis.model_dump(),
            "entities": entity_analysis.model_dump(),
            "grammar": grammar_analysis.model_dump() if grammar_analysis else None,
            "experience": experience_analysis.model_dump() if experience_analysis else None,
            "processing_time_ms": round(processing_time_ms, 2),
        }

        logger.info(
            f"Анализ завершён для resume_id {request.resume_id} за {processing_time_ms:.2f}мс"
        )

        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content=response_data,
        )

    except HTTPException:
        # Перебрасываем HTTP-исключения
        raise
    except Exception as e:
        logger.error(f"Ошибка анализа резюме: {e}", exc_info=True)
        error_msg = get_error_message("analysis_failed", locale)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=error_msg,
        ) from e


@router.get(
    "/{resume_id}",
    response_model=dict,
    status_code=status.HTTP_200_OK,
    tags=["Analysis"],
)
async def get_analysis_result(
    http_request: Request, resume_id: str
) -> JSONResponse:
    """
    Получить результат анализа для конкретного резюме.

    Этот эндпоинт возвращает результаты анализа резюме.
    Сейчас возвращает заглушки, так как полная интеграция с БД ожидается.

    Args:
        http_request: Объект запроса FastAPI (для заголовка Accept-Language)
        resume_id: ID резюме для получения анализа

    Returns:
        JSON-ответ с результатами анализа

    Raises:
        HTTPException(404): Если резюме не найдено

    Examples:
        >>> import requests
        >>> response = requests.get("http://localhost:8000/api/resumes/abc123")
        >>> response.json()
        {
            "resume_id": "abc123",
            "status": "pending",
            "errors": [],
            "grammar_errors": [],
            "keywords": [],
            "technical_skills": []
        }
    """
    locale = _extract_locale(http_request)
    logger.info(f"Получение анализа для resume_id: {resume_id}")

    # TODO: Реализовать поиск в базе данных в следующей подзадаче
    # Пока возвращаем заглушку с правильной структурой
    return JSONResponse(
        status_code=status.HTTP_200_OK,
        content={
            "resume_id": resume_id,
            "status": "pending",
            "message": "Анализ не найден - пожалуйста, сначала запустите анализ",
            "errors": [],
            "grammar_errors": [],
            "keywords": [],
            "technical_skills": [],
            "total_experience_months": 0,
            "matched_skills": [],
            "missing_skills": [],
            "match_percentage": 0,
        },
    )
