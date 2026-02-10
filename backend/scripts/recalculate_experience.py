"""
Скрипт для пересчёта опыта работы для всех резюме в базе данных.

Этот скрипт:
1. Загружает все резюме из базы данных
2. Для каждого резюме извлекает текст из файла
3. Рассчитывает общий опыт работы в месяцах
4. Сохраняет результат в базу данных
"""
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from analyzers.experience_calculator import calculate_total_experience_from_text
from database import get_db
from models.resume import Resume
from services.data_extractor.extract import extract_text_from_pdf, extract_text_from_docx
from sqlalchemy import select
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def recalculate_experience_for_all():
    """Пересчитать опыт для всех резюме в базе данных."""
    from database import async_session_maker

    async with async_session_maker() as db:
        try:
            # Получить все резюме
            result = await db.execute(select(Resume))
            resumes = result.scalars().all()

            logger.info(f"Найдено {len(resumes)} резюме для обработки")

            updated_count = 0
            skipped_count = 0

            for resume in resumes:
                try:
                    # Пропустить, если опыт уже рассчитан
                    if resume.total_experience_months is not None and resume.total_experience_months > 0:
                        logger.info(f"Резюме {resume.id} уже имеет опыт: {resume.total_experience_months} месяцев")
                        skipped_count += 1
                        continue

                    # Извлечь текст из файла
                    file_path = Path(resume.file_path)
                    if not file_path.exists():
                        logger.warning(f"Файл не найден: {file_path}")
                        skipped_count += 1
                        continue

                    # Извлечение текста в зависимости от типа файла
                    if file_path.suffix.lower() == '.pdf':
                        extract_result = extract_text_from_pdf(str(file_path))
                    elif file_path.suffix.lower() == '.docx':
                        extract_result = extract_text_from_docx(str(file_path))
                    else:
                        logger.warning(f"Неподдерживаемый тип файла: {file_path.suffix}")
                        skipped_count += 1
                        continue

                    text = extract_result.get('text', '')
                    if not text or len(text.strip()) < 10:
                        logger.warning(f"Не удалось извлечь текст из {file_path}")
                        skipped_count += 1
                        continue

                    # Рассчитать опыт
                    experience_data = calculate_total_experience_from_text(text)
                    total_months = experience_data.get('total_months', 0)

                    if total_months > 0:
                        # Сохранить в базу данных
                        resume.total_experience_months = total_months
                        updated_count += 1

                        years = total_months // 12
                        months = total_months % 12
                        logger.info(
                            f"Обновлено резюме {resume.id}: {years} лет {months} месяцев "
                            f"({resume.filename})"
                        )
                    else:
                        logger.info(f"Опыт не найден в резюме {resume.id} ({resume.filename})")
                        # Сохранить 0 если опыт не найден
                        resume.total_experience_months = 0

                except Exception as e:
                    logger.error(f"Ошибка обработки резюме {resume.id}: {e}", exc_info=True)
                    skipped_count += 1
                    continue

            # Сохранить все изменения
            await db.commit()

            logger.info(f"\n{'='*60}")
            logger.info(f"Обработка завершена!")
            logger.info(f"Обновлено: {updated_count} резюме")
            logger.info(f"Пропущено: {skipped_count} резюме")
            logger.info(f"{'='*60}")

        except Exception as e:
            logger.error(f"Критическая ошибка: {e}", exc_info=True)
            await db.rollback()


if __name__ == "__main__":
    asyncio.run(recalculate_experience_for_all())
