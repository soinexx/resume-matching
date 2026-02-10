"""
Модель Resume для хранения данных загруженных резюме
"""
import enum
from typing import Optional

from sqlalchemy import Enum, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from .base import Base, TimestampMixin, UUIDMixin


class ResumeStatus(str, enum.Enum):
    """Статус обработки резюме"""

    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class Resume(Base, UUIDMixin, TimestampMixin):
    """
    Модель Resume для хранения загруженных файлов резюме и метаданных

    Attributes:
        id: Первичный ключ UUID
        filename: Исходное имя файла загруженного резюме
        file_path: Путь к сохранённому файлу резюме
        content_type: MIME-тип файла (например, application/pdf)
        status: Текущий статус обработки
        raw_text: Извлеченный текстовый контент из резюме
        language: Обнаруженный язык (en, ru и т.д.)
        error_message: Сообщение об ошибке, если обработка не удалась
        uploaded_at: Временная метка загрузки резюме (унаследовано от TimestampMixin)
    """

    __tablename__ = "resumes"

    filename: Mapped[str] = mapped_column(String(255), nullable=False)
    file_path: Mapped[str] = mapped_column(String(512), nullable=False)
    content_type: Mapped[str] = mapped_column(String(100), nullable=False)
    status: Mapped[ResumeStatus] = mapped_column(
        Enum(ResumeStatus), default=ResumeStatus.PENDING, nullable=False, index=True
    )
    raw_text: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    language: Mapped[Optional[str]] = mapped_column(String(10), nullable=True)
    total_experience_months: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    error_message: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    def __repr__(self) -> str:
        return f"<Resume(id={self.id}, filename={self.filename}, status={self.status.value})>"
