"""
Единый сопоставитель навыков, комбинирующий несколько стратегий сопоставления.

Этот модуль предоставляет комплексный сопоставитель, который объединяет:
1. Улучшенное сопоставление ключевых слов (синонимы, нечёткое, составные навыки)
2. Взвешенное сопоставление TF-IDF (оценка на основе важности)
3. Векторное семантическое сходство (sentence-transformers)

Единый подход объединяет лучшие возможности всех методов:
- Точное сопоставление ключевых слов
- Взвешенная оценка важности
- Семантическое понимание
"""
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .enhanced_matcher import EnhancedSkillMatcher
from .tfidf_matcher import TfidfSkillMatcher, TfidfMatchResult
from .vector_matcher import VectorSimilarityMatcher, VectorMatchResult, _HAS_SENTENCE_TRANSFORMERS

logger = logging.getLogger(__name__)


@dataclass
class UnifiedMatchResult:
    """Комплексный результат единого сопоставления."""

    # Общие оценки
    overall_score: float
    passed: bool

    # Результаты отдельных методов
    keyword_score: float
    keyword_passed: bool
    keyword_matches: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    tfidf_score: float = 0.0
    tfidf_passed: bool = False
    tfidf_matched: List[str] = field(default_factory=list)
    tfidf_missing: List[str] = field(default_factory=list)

    vector_score: float = 0.0
    vector_passed: bool = False
    vector_similarity: float = 0.0

    # Детальная информация о совпадениях
    matched_skills: List[str] = field(default_factory=list)
    missing_skills: List[str] = field(default_factory=list)

    # Рекомендация
    recommendation: str = "neutral"

    def to_dict(self) -> Dict[str, Any]:
        """Преобразовать результат в словарь для JSON-сериализации."""
        return {
            "overall_score": self.overall_score,
            "passed": self.passed,
            "keyword_score": self.keyword_score,
            "keyword_passed": self.keyword_passed,
            "tfidf_score": self.tfidf_score,
            "tfidf_passed": self.tfidf_passed,
            "tfidf_matched": self.tfidf_matched,
            "tfidf_missing": self.tfidf_missing,
            "vector_score": self.vector_score,
            "vector_passed": self.vector_passed,
            "vector_similarity": self.vector_similarity,
            "matched_skills": self.matched_skills,
            "missing_skills": self.missing_skills,
            "recommendation": self.recommendation,
        }


class UnifiedSkillMatcher:
    """
    Единый сопоставитель навыков, комбинирующий несколько стратегий.

    Использует три взаимодополняющих подхода:
    1. Улучшенное сопоставление: прямое, синонимы, нечёткое, составных навыков (вес 80%)
    2. Сопоставление TF-IDF: взвешенная оценка на основе важности ключевых слов (вес 15%)
    3. Векторное сопоставление: семантическое сходство с использованием эмбеддингов (вес 5%)

    Общая оценка является взвешенной комбинацией всех трёх методов.
    По умолчанию приоритет отдаётся прямому сопоставлению навыков для более
    прозрачных и предсказуемых результатов.

    Example:
        >>> matcher = UnifiedSkillMatcher()
        >>> result = matcher.match(
        ...     resume_text="Опыт работы с React и Python",
        ...     resume_skills=["React", "Python"],
        ...     job_title="Senior React Developer",
        ...     job_description="Ищем эксперта по React",
        ...     required_skills=["React", "Python", "TypeScript"]
        ... )
        >>> print(result.overall_score)
        0.75
        >>> print(result.recommendation)
        'good_match'
    """

    def __init__(
        self,
        # Веса для общей оценки (должны суммироваться до 1.0)
        keyword_weight: float = 0.8,
        tfidf_weight: float = 0.15,
        vector_weight: float = 0.05,

        # Пороги отдельных методов
        keyword_threshold: Optional[float] = None,  # Использует EnhancedSkillMatcher по умолчанию
        tfidf_threshold: float = 0.3,
        vector_threshold: float = 0.5,

        # Общий порог прохождения
        overall_threshold: float = 0.5,

        # Конфигурация модели
        vector_model: str = "all-MiniLM-L6-v2",
    ):
        """
        Инициализировать единый сопоставитель навыков.

        Args:
            keyword_weight: Вес сопоставления ключевых слов в общей оценке (по умолчанию: 0.8)
            tfidf_weight: Вес сопоставления TF-IDF в общей оценке (по умолчанию: 0.15)
            vector_weight: Вес векторного сопоставления в общей оценке (по умолчанию: 0.05)
            keyword_threshold: Порог для прохождения/отказа сопоставления ключевых слов
            tfidf_threshold: Порог для прохождения/отказа сопоставления TF-IDF
            vector_threshold: Порог для прохождения/отказа векторного сопоставления
            overall_threshold: Порог для общего прохождения/отказа
            vector_model: Название модели sentence-transformers

        Note:
            По умолчанию приоритет отдаётся ключевому сопоставлению (80%) для более
            прозрачных результатов. Веса будут нормализованы, чтобы в сумме давать 1.0.
        """
        # Нормализовать веса
        total_weight = keyword_weight + tfidf_weight + vector_weight
        self.keyword_weight = keyword_weight / total_weight
        self.tfidf_weight = tfidf_weight / total_weight
        self.vector_weight = vector_weight / total_weight

        self.tfidf_threshold = tfidf_threshold
        self.vector_threshold = vector_threshold
        self.overall_threshold = overall_threshold

        # Инициализировать сопоставители
        self.keyword_matcher = EnhancedSkillMatcher()
        self.tfidf_matcher = TfidfSkillMatcher(threshold=tfidf_threshold)

        if _HAS_SENTENCE_TRANSFORMERS:
            self.vector_matcher = VectorSimilarityMatcher(
                threshold=vector_threshold,
                model_name=vector_model,
            )
        else:
            self.vector_matcher = None
            logger.warning("Векторное сопоставление отключено (sentence-transformers недоступен)")

        logger.info(
            f"UnifiedSkillMatcher инициализирован с весами: "
            f"keyword={self.keyword_weight:.2f}, "
            f"tfidf={self.tfidf_weight:.2f}, "
            f"vector={self.vector_weight:.2f}"
        )

    def match(
        self,
        resume_text: str,
        resume_skills: List[str],
        job_title: str,
        job_description: str,
        required_skills: List[str],
        context: Optional[str] = None,
    ) -> UnifiedMatchResult:
        """
        Perform unified matching using all three methods.

        Args:
            resume_text: Full resume text
            resume_skills: List of skills extracted from resume
            job_title: Job posting title
            job_description: Job posting description
            required_skills: List of required skills from job posting
            context: Optional context hint for keyword matching

        Returns:
            UnifiedMatchResult with comprehensive match information
        """
        # 1. Enhanced keyword matching
        keyword_results = self.keyword_matcher.match_multiple(
            resume_skills=resume_skills,
            required_skills=required_skills,
            context=context,
        )

        matched_skills = [
            skill for skill, result in keyword_results.items()
            if result.get("matched", False)
        ]
        missing_skills = [
            skill for skill, result in keyword_results.items()
            if not result.get("matched", False)
        ]

        keyword_pct = self.keyword_matcher.calculate_match_percentage(keyword_results)
        keyword_score = keyword_pct / 100
        keyword_passed = keyword_score >= 0.3  # At least 30% skill match

        # 2. TF-IDF matching
        tfidf_result = self.tfidf_matcher.match(
            resume_text=resume_text,
            job_title=job_title,
            job_description=job_description,
            required_skills=required_skills,
        )

        # 3. Vector matching
        vector_score = 0.0
        vector_passed = False
        vector_similarity = 0.0

        if self.vector_matcher:
            vector_result = self.vector_matcher.match(
                resume_text=resume_text,
                job_title=job_title,
                job_description=job_description,
                required_skills=required_skills,
            )
            vector_score = vector_result.score
            vector_passed = vector_result.passed
            vector_similarity = vector_result.similarity

        # Calculate overall score (weighted combination)
        overall_score = (
            self.keyword_weight * keyword_score +
            self.tfidf_weight * tfidf_result.score +
            self.vector_weight * vector_score
        )

        overall_passed = overall_score >= self.overall_threshold

        # Generate recommendation
        recommendation = self._generate_recommendation(
            overall_score, keyword_passed, tfidf_result.passed, vector_passed
        )

        return UnifiedMatchResult(
            overall_score=round(overall_score, 3),
            passed=overall_passed,
            keyword_score=round(keyword_score, 3),
            keyword_passed=keyword_passed,
            keyword_matches=keyword_results,
            tfidf_score=round(tfidf_result.score, 3),
            tfidf_passed=tfidf_result.passed,
            tfidf_matched=tfidf_result.matched_keywords,
            tfidf_missing=tfidf_result.missing_keywords,
            vector_score=round(vector_score, 3),
            vector_passed=vector_passed,
            vector_similarity=round(vector_similarity, 3),
            matched_skills=matched_skills,
            missing_skills=missing_skills,
            recommendation=recommendation,
        )

    def _generate_recommendation(
        self,
        overall_score: float,
        keyword_passed: bool,
        tfidf_passed: bool,
        vector_passed: bool,
    ) -> str:
        """
        Generate hiring recommendation based on match results.

        Args:
            overall_score: Overall match score
            keyword_passed: Whether keyword matching passed
            tfidf_passed: Whether TF-IDF matching passed
            vector_passed: Whether vector matching passed

        Returns:
            Recommendation string: 'excellent', 'good', 'maybe', 'poor'
        """
        if overall_score >= 0.8 and keyword_passed:
            return "excellent"
        elif overall_score >= 0.6 and keyword_passed and tfidf_passed:
            return "good"
        elif overall_score >= 0.4:
            return "maybe"
        else:
            return "poor"

    def match_resume_to_vacancy(
        self,
        resume_text: str,
        resume_skills: List[str],
        vacancy_title: str,
        vacancy_description: str,
        vacancy_skills: List[str],
    ) -> UnifiedMatchResult:
        """
        Match a resume to a specific vacancy.

        Convenience method with clearer naming for vacancy matching.

        Args:
            resume_text: Full resume text
            resume_skills: Extracted skills from resume
            vacancy_title: Vacancy title
            vacancy_description: Vacancy description
            vacancy_skills: Required skills for vacancy

        Returns:
            UnifiedMatchResult with comprehensive match information
        """
        return self.match(
            resume_text=resume_text,
            resume_skills=resume_skills,
            job_title=vacancy_title,
            job_description=vacancy_description,
            required_skills=vacancy_skills,
        )

    def rank_candidates(
        self,
        candidates: List[Dict[str, Any]],
        job_title: str,
        job_description: str,
        required_skills: List[str],
    ) -> List[Dict[str, Any]]:
        """
        Rank multiple candidates for a job posting.

        Args:
            candidates: List of candidate dicts with 'resume_text', 'resume_skills', 'id', 'name'
            job_title: Job posting title
            job_description: Job posting description
            required_skills: List of required skills

        Returns:
            List of candidates with added 'match_result' field, sorted by overall_score
        """
        results = []

        for candidate in candidates:
            match_result = self.match(
                resume_text=candidate.get("resume_text", ""),
                resume_skills=candidate.get("resume_skills", []),
                job_title=job_title,
                job_description=job_description,
                required_skills=required_skills,
            )

            results.append({
                **candidate,
                "match_result": match_result.to_dict(),
                "overall_score": match_result.overall_score,
            })

        # Sort by overall score descending
        results.sort(key=lambda x: x["overall_score"], reverse=True)
        return results


# Singleton instance
_default_matcher: Optional[UnifiedSkillMatcher] = None


def get_unified_matcher() -> UnifiedSkillMatcher:
    """Get or create default unified matcher instance."""
    global _default_matcher
    if _default_matcher is None:
        _default_matcher = UnifiedSkillMatcher()
    return _default_matcher
