"""
Сопоставитель навыков на основе TF-IDF со взвешенной оценкой ключевых слов.

Этот модуль обеспечивает интеллектуальное сопоставление навыков с использованием TF-IDF
(частотность термина - обратная частотность документа) для ранжирования важности
ключевых слов и предоставления взвешенных оценок сопоставления.

Основные возможности:
- Взвешенная оценка на основе важности термина
- Отсутствующие ключевые слова ранжируются по важности TF-IDF
- Поддержка N-грам (1-2 граммы) для сопоставления фраз
- Настраиваемые пороги и ограничения функций
"""
import logging
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from sklearn.feature_extraction.text import TfidfVectorizer

logger = logging.getLogger(__name__)


@dataclass
class TfidfMatchResult:
    """Результат сопоставления на основе TF-IDF."""

    score: float
    passed: bool
    matched_keywords: List[str]
    missing_keywords: List[str]
    keyword_weights: Dict[str, float]


class TfidfSkillMatcher:
    """
    Сопоставитель навыков на основе TF-IDF со взвешенной оценкой.

    Использует TF-IDF для расчёта важности ключевых слов в вакансии
    и сопоставляет их с текстом резюме. Предоставляет взвешенные оценки,
    отражающие относительную важность каждого ключевого слова.

    Example:
        >>> matcher = TfidfSkillMatcher()
        >>> result = matcher.match(
        ...     resume_text="Опыт работы с React и Python",
        ...     job_title="Senior React Developer",
        ...     job_description="Ищем эксперта по React",
        ...     required_skills=["React", "Python", "TypeScript"]
        ... )
        >>> print(result.score)
        0.67
        >>> print(result.missing_keywords)
        ['TypeScript']
    """

    def __init__(
        self,
        threshold: float = 0.3,
        max_features: int = 100,
        tfidf_cutoff: float = 0.1,  # Увеличено с 0.05 для фильтрации очень редких слов
        max_missing_display: int = 10,
        use_log_weights: bool = True,  # Использовать логарифмическое сглаживание весов
    ):
        """
        Инициализировать сопоставитель навыков TF-IDF.

        Args:
            threshold: Минимальная оценка для прохождения (0.0-1.0)
            max_features: Максимальное количество признаков TF-IDF
            tfidf_cutoff: Минимальная оценка TF-IDF для значимого ключевого слова (по умолчанию: 0.1)
            max_missing_display: Максимальное количество отсутствующих ключевых слов для возврата
            use_log_weights: Использовать логарифмическое сглаживание для уменьшения влияния редких слов
        """
        self.threshold = threshold
        self.max_features = max_features
        self.tfidf_cutoff = tfidf_cutoff
        self.max_missing_display = max_missing_display
        self.use_log_weights = use_log_weights

    def _create_vectorizer(self) -> TfidfVectorizer:
        """
        Создать векторизатор TF-IDF с оптимизированными настройками.

        Returns:
            Настроенный экземпляр TfidfVectorizer
        """
        return TfidfVectorizer(
            stop_words="english",
            ngram_range=(1, 2),  # Включить биграммы для фраз типа "machine learning"
            max_features=self.max_features,
            token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z0-9+#.-]*\b",
            lowercase=True,
        )

    def _extract_keywords_from_job(
        self,
        job_title: str,
        job_description: str,
        required_skills: List[str],
    ) -> Tuple[List[str], Dict[str, float]]:
        """
        Извлечь значимые ключевые слова из вакансии с использованием TF-IDF.

        Args:
            job_title: Название вакансии
            job_description: Описание вакансии
            required_skills: Список требуемых навыков

        Returns:
            Кортеж (significant_keywords, tfidf_scores)
        """
        # Объединить текст вакансии
        job_text = f"{job_title} {job_description} {' '.join(required_skills)}"

        if not job_text.strip():
            return [], {}

        # Создать и настроить векторизатор
        vectorizer = self._create_vectorizer()

        try:
            vectorizer.fit([job_text.lower()])
        except ValueError:
            logger.warning("Не удалось настроить TF-IDF векторизатор на тексте вакансии")
            return required_skills, {kw: 0.1 for kw in required_skills}

        # Получить оценки TF-IDF
        feature_names = vectorizer.get_feature_names_out()
        tfidf_matrix = vectorizer.transform([job_text.lower()])
        tfidf_scores = dict(zip(feature_names, tfidf_matrix.toarray()[0]))

        # Извлечь значимые ключевые слова
        significant_keywords = {
            term for term, score in tfidf_scores.items()
            if score > self.tfidf_cutoff
        }

        # Всегда включать явно требуемые навыки
        for skill in required_skills:
            significant_keywords.add(skill.lower())

        return list(significant_keywords), tfidf_scores

    def _find_keyword_matches(
        self,
        resume_text: str,
        keywords: List[str],
    ) -> Tuple[List[str], List[str]]:
        """
        Найти, какие ключевые слова присутствуют в тексте резюме.

        Args:
            resume_text: Текст резюме в нижнем регистре
            keywords: Список ключевых слов для поиска

        Returns:
            Кортеж (matched_keywords, missing_keywords)
        """
        matched = []
        missing = []

        for keyword in keywords:
            # Использовать шаблон границ слов для точного сопоставления
            pattern = rf"\b{re.escape(keyword)}\b"
            if re.search(pattern, resume_text):
                matched.append(keyword)
            else:
                missing.append(keyword)

        return matched, missing

    def match(
        self,
        resume_text: str,
        job_title: str,
        job_description: str,
        required_skills: List[str],
        threshold: Optional[float] = None,
    ) -> TfidfMatchResult:
        """
        Сопоставить резюме с вакансией с использованием взвешенной оценки TF-IDF.

        Args:
            resume_text: Содержание текста резюме
            job_title: Название вакансии
            job_description: Описание вакансии
            required_skills: Список требуемых навыков из вакансии
            threshold: Переопределить порог по умолчанию

        Returns:
            TfidfMatchResult с оценкой, статусом прохождения и деталями ключевых слов
        """
        if threshold is None:
            threshold = self.threshold

        # Нормализовать текст резюме
        resume_lower = resume_text.lower()

        # Извлечь значимые ключевые слова из вакансии
        keywords, tfidf_scores = self._extract_keywords_from_job(
            job_title, job_description, required_skills
        )

        if not keywords:
            # Нет ключевых слов для сопоставления
            return TfidfMatchResult(
                score=1.0,
                passed=True,
                matched_keywords=[],
                missing_keywords=[],
                keyword_weights={},
            )

        # Найти совпадения
        matched, missing = self._find_keyword_matches(resume_lower, keywords)

        # Рассчитать взвешенную оценку с логарифмическим сглаживанием
        # Логарифм уменьшает влияние редких слов с экстремально высоким TF-IDF score
        import math

        def smooth_weight(weight: float) -> float:
            """Сгладить вес, используя логарифм для уменьшения влияния редких слов."""
            if self.use_log_weights:
                # log(1 + weight*10) сглаживает веса от 0 до ~3
                return math.log(1 + weight * 10)
            return weight

        matched_weight = sum(smooth_weight(tfidf_scores.get(kw, 0.05)) for kw in matched)
        total_weight = sum(smooth_weight(tfidf_scores.get(kw, 0.05)) for kw in keywords)
        score = matched_weight / total_weight if total_weight > 0 else 1.0

        # Сортировать отсутствующие по важности TF-IDF (сначала наиболее важные)
        missing_sorted = sorted(
            missing,
            key=lambda kw: tfidf_scores.get(kw, 0),
            reverse=True,
        )[:self.max_missing_display]

        # Построить веса ключевых слов для совпавших
        keyword_weights = {kw: tfidf_scores.get(kw, 0.1) for kw in matched}

        return TfidfMatchResult(
            score=float(score),
            passed=bool(score >= threshold),
            matched_keywords=matched,
            missing_keywords=missing_sorted,
            keyword_weights=keyword_weights,
        )

    def match_resume_to_vacancy(
        self,
        resume_text: str,
        resume_skills: List[str],
        vacancy_title: str,
        vacancy_description: str,
        vacancy_skills: List[str],
    ) -> TfidfMatchResult:
        """
        Сопоставить резюме с конкретной вакансией.

        Удобный метод, который объединяет навыки резюме с полным текстом
        для комплексного сопоставления.

        Args:
            resume_text: Полный текст резюме
            resume_skills: Извлечённые навыки из резюме
            vacancy_title: Название вакансии
            vacancy_description: Описание вакансии
            vacancy_skills: Требуемые навыки для вакансии

        Returns:
            TfidfMatchResult с деталями сопоставления
        """
        # Объединить текст резюме с навыками для лучшего сопоставления
        enhanced_resume = f"{resume_text} {' '.join(resume_skills)}"

        return self.match(
            resume_text=enhanced_resume,
            job_title=vacancy_title,
            job_description=vacancy_description,
            required_skills=vacancy_skills,
        )

    def get_missing_importance(
        self,
        result: TfidfMatchResult,
        top_n: int = 5,
    ) -> List[Tuple[str, float]]:
        """
        Получить топ отсутствующих ключевых слов по важности.

        Args:
            result: TfidfMatchResult из match()
            top_n: Количество топ ключевых слов для возврата

        Returns:
            Список кортежей (keyword, importance_score)
        """
        # Это потребовало бы хранения tfidf_scores в результате
        # Пока что, вернуть отсутствующие ключевые слова
        return [(kw, 0.0) for kw in result.missing_keywords[:top_n]]


# Singleton instance for convenience
_default_matcher: Optional[TfidfSkillMatcher] = None


def get_tfidf_matcher() -> TfidfSkillMatcher:
    """Get or create default TF-IDF matcher instance."""
    global _default_matcher
    if _default_matcher is None:
        _default_matcher = TfidfSkillMatcher()
    return _default_matcher
