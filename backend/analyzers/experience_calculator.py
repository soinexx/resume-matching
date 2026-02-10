"""
Калькулятор опыта для истории работы резюме.

Этот модуль предоставляет функции для расчёта общего рабочего опыта из данных резюме,
включая обработку перекрывающихся периодов, фильтрацию по навыкам и конвертацию
между месяцами и годами.
"""
import logging
import re
from datetime import datetime
from typing import Dict, List, Optional, Union

logger = logging.getLogger(__name__)


def _parse_date(date_str: Optional[str]) -> Optional[datetime]:
    """
    Парсить строку даты в различных форматах.

    Поддерживаемые форматы:
    - ГГГГ-ММ-ДД (формат ISO)
    - ММ/ГГГГ
    - ГГГГ-ММ
    - Месяц ГГГГ (например, "May 2020")
    - None (возвращает None, указывает на текущую дату)

    Args:
        date_str: Строка даты для парсинга или None

    Returns:
        Объект datetime или None, если date_str равно None

    Raises:
        ValueError: Если строка даты не может быть распаршена

    Examples:
        >>> _parse_date("2023-02-01")
        datetime.datetime(2023, 2, 1, 0, 0)
        >>> _parse_date("02/2023")
        datetime.datetime(2023, 2, 1, 0, 0)
        >>> _parse_date(None)  # Возвращает None (текущая дата)
    """
    if date_str is None:
        return None

    if not isinstance(date_str, str):
        raise ValueError(f"Date must be string or None, got {type(date_str)}")

    date_str = date_str.strip()

    # Расширенный список форматов даты для попытки
    formats = [
        "%Y-%m-%d",     # 2023-02-01
        "%Y-%m",        # 2023-02
        "%m/%Y",        # 02/2023
        "%m.%Y",        # 02.2023
        "%m %Y",        # 02 2025, 06 2025
        "%b %Y",        # Feb 2023, Jun 2025
        "%B %Y",        # February 2023, June 2025
        "%Y/%m",        # 2023/02
        "%Y.%m",        # 2023.02
    ]

    for fmt in formats:
        try:
            return datetime.strptime(date_str, fmt)
        except ValueError:
            continue

    raise ValueError(f"Unable to parse date: {date_str}")


def _calculate_months_between(
    start_date: datetime, end_date: Optional[datetime]
) -> int:
    """
    Рассчитать количество месяцев между двумя датами.

    Args:
        start_date: Начальная дата (включительно)
        end_date: Конечная дата (включительно) или None для текущей даты

    Returns:
        Количество месяцев между датами (минимум 1 месяц)

    Examples:
        >>> from datetime import datetime
        >>> start = datetime(2023, 2, 1)
        >>> end = datetime(2023, 5, 1)
        >>> _calculate_months_between(start, end)
        3
    """
    if end_date is None:
        end_date = datetime.now()

    # Рассчитать разницу в месяцах
    months = (end_date.year - start_date.year) * 12 + (
        end_date.month - start_date.month
    )

    # Добавить частичный месяц, если дни указывают на больше полных месяцев
    if end_date.day >= start_date.day:
        months += 1

    return max(1, months)  # Minimum 1 month


def _dates_overlap(
    period1_start: datetime,
    period1_end: Optional[datetime],
    period2_start: datetime,
    period2_end: Optional[datetime],
) -> bool:
    """
    Check if two date periods overlap.

    Args:
        period1_start: Start date of period 1
        period1_end: End date of period 1 (None means current date)
        period2_start: Start date of period 2
        period2_end: End date of period 2 (None means current date)

    Returns:
        True if periods overlap, False otherwise

    Examples:
        >>> from datetime import datetime
        >>> p1_start = datetime(2020, 1, 1)
        >>> p1_end = datetime(2020, 6, 1)
        >>> p2_start = datetime(2020, 5, 1)
        >>> p2_end = datetime(2020, 12, 1)
        >>> _dates_overlap(p1_start, p1_end, p2_start, p2_end)
        True
    """
    if period1_end is None:
        period1_end = datetime.now()
    if period2_end is None:
        period2_end = datetime.now()

    # Check for overlap: max(start1, start2) <= min(end1, end2)
    latest_start = max(period1_start, period2_start)
    earliest_end = min(period1_end, period2_end)

    return latest_start <= earliest_end


def calculate_total_experience(
    experience: List[Dict[str, Optional[str]]],
    *,
    handle_overlaps: bool = True,
    skip_invalid: bool = True,
) -> Dict[str, Optional[Union[int, str, List[Dict]]]]:
    """
    Calculate total work experience in months from a list of work periods.

    This function processes a list of work experience entries, each containing
    start and end dates, and calculates the total experience in months.
    It can handle overlapping periods to avoid double-counting.

    Args:
        experience: List of experience dictionaries, each containing:
            - start: Start date string (required)
            - end: End date string or None if current position (optional)
            - company: Company name (optional, for logging)
            - position: Job title (optional, for logging)
            - description: Job description (optional)
        handle_overlaps: If True, merge overlapping periods to avoid double-counting
        skip_invalid: If True, skip entries with invalid dates; if False, raise error

    Returns:
        Dictionary containing:
            - total_months: Total experience in months
            - total_years: Total experience in years (float)
            - periods: List of processed period information
            - overlap_count: Number of overlapping periods detected (if handle_overlaps=True)
            - error: Error message if calculation failed

    Raises:
        ValueError: If skip_invalid=False and entry has invalid dates

    Examples:
        >>> experience = [
        ...     {"start": "2020-05-01", "end": "2023-02-01", "company": "Company A"},
        ...     {"start": "2023-02-01", "end": None, "company": "Company B"},
        ... ]
        >>> result = calculate_total_experience(experience)
        >>> result["total_months"]
        46
    """
    if not experience:
        return {
            "total_months": 0,
            "total_years": 0.0,
            "periods": [],
            "overlap_count": 0,
        }

    try:
        periods = []
        overlap_count = 0

        for idx, entry in enumerate(experience):
            if not isinstance(entry, dict):
                if skip_invalid:
                    logger.warning(f"Skipping invalid entry {idx}: not a dictionary")
                    continue
                else:
                    raise ValueError(f"Entry {idx} is not a dictionary")

            start_str = entry.get("start")
            end_str = entry.get("end")

            if not start_str:
                if skip_invalid:
                    logger.warning(f"Skipping entry {idx}: missing start date")
                    continue
                else:
                    raise ValueError(f"Entry {idx} missing start date")

            try:
                start_date = _parse_date(start_str)
                end_date = _parse_date(end_str) if end_str else None
            except ValueError as e:
                if skip_invalid:
                    logger.warning(f"Skipping entry {idx}: {e}")
                    continue
                else:
                    raise ValueError(f"Entry {idx} has invalid date: {e}") from e

            months = _calculate_months_between(start_date, end_date)

            period_info = {
                "index": idx,
                "company": entry.get("company"),
                "position": entry.get("position"),
                "start": start_str,
                "end": end_str,
                "start_parsed": start_date.isoformat(),
                "end_parsed": end_date.isoformat() if end_date else None,
                "months": months,
            }
            periods.append(period_info)

        # Handle overlaps if requested
        if handle_overlaps and len(periods) > 1:
            merged_periods = _merge_overlapping_periods(periods)
            overlap_count = len(periods) - len(merged_periods)
            periods = merged_periods

        # Calculate total
        total_months = sum(p["months"] for p in periods)
        total_years = round(total_months / 12, 2)

        return {
            "total_months": total_months,
            "total_years": total_years,
            "periods": periods,
            "overlap_count": overlap_count if handle_overlaps else 0,
        }

    except Exception as e:
        logger.error(f"Error calculating total experience: {e}")
        return {
            "total_months": None,
            "total_years": None,
            "periods": [],
            "overlap_count": 0,
            "error": str(e),
        }


def _merge_overlapping_periods(
    periods: List[Dict],
) -> List[Dict]:
    """
    Merge overlapping periods to avoid double-counting months.

    Args:
        periods: List of period dictionaries with start_parsed, end_parsed, months

    Returns:
        List of merged periods without overlaps

    Examples:
        >>> periods = [
        ...     {"start_parsed": "2020-01-01", "end_parsed": "2020-06-01", "months": 6},
        ...     {"start_parsed": "2020-05-01", "end_parsed": "2020-12-01", "months": 8},
        ... ]
        >>> merged = _merge_overlapping_periods(periods)
        >>> len(merged)
        1
    """
    if not periods:
        return []

    # Sort by start date
    sorted_periods = sorted(
        periods,
        key=lambda p: datetime.fromisoformat(p["start_parsed"]),
    )

    merged = []
    current = sorted_periods[0].copy()

    for period in sorted_periods[1:]:
        current_start = datetime.fromisoformat(current["start_parsed"])
        current_end = datetime.fromisoformat(
            current["end_parsed"]
            if current["end_parsed"]
            else datetime.now().isoformat()
        )
        period_start = datetime.fromisoformat(period["start_parsed"])
        period_end = datetime.fromisoformat(
            period["end_parsed"]
            if period["end_parsed"]
            else datetime.now().isoformat()
        )

        # Check if periods overlap
        if _dates_overlap(current_start, current_end, period_start, period_end):
            # Merge periods
            merged_start = min(current_start, period_start)
            merged_end = max(current_end, period_end)

            current["start_parsed"] = merged_start.isoformat()
            current["end_parsed"] = merged_end.isoformat()
            current["months"] = _calculate_months_between(merged_start, merged_end)
        else:
            # No overlap, add current and start new
            merged.append(current)
            current = period.copy()

    merged.append(current)
    return merged


def calculate_skill_experience(
    experience: List[Dict[str, Optional[str]]],
    skill: str,
    *,
    case_sensitive: bool = False,
    handle_overlaps: bool = True,
) -> Dict[str, Optional[Union[int, str, List[Dict]]]]:
    """
    Calculate total experience with a specific skill across all projects.

    This function filters work experience entries where the specified skill
    is mentioned in the job description or position, then calculates the
    total duration in months.

    Args:
        experience: List of experience dictionaries
        skill: Skill name to search for (e.g., "Java", "Python", "React")
        case_sensitive: Whether skill matching should be case-sensitive
        handle_overlaps: If True, merge overlapping periods to avoid double-counting

    Returns:
        Dictionary containing:
            - skill: The skill searched for
            - total_months: Total months of experience with this skill
            - total_years: Total years of experience with this skill
            - matching_projects: List of projects where skill was found
            - projects_count: Number of projects where skill was found
            - error: Error message if calculation failed

    Examples:
        >>> experience = [
        ...     {
        ...         "start": "2020-05-01",
        ...         "end": "2023-02-01",
        ...         "description": "Java development using Spring Boot"
        ...     },
        ...     {
        ...         "start": "2023-02-01",
        ...         "end": None,
        ...         "description": "React frontend development"
        ...     },
        ... ]
        >>> result = calculate_skill_experience(experience, "Java")
        >>> result["total_months"]
        33
    """
    if not skill:
        return {
            "skill": skill,
            "total_months": 0,
            "total_years": 0.0,
            "matching_projects": [],
            "projects_count": 0,
            "error": "Skill cannot be empty",
        }

    try:
        matching_projects = []
        search_skill = skill if case_sensitive else skill.lower()

        for idx, entry in enumerate(experience):
            if not isinstance(entry, dict):
                continue

            # Search in description and position
            description = entry.get("description", "") or ""
            position = entry.get("position", "") or ""

            if case_sensitive:
                text = description + " " + position
            else:
                text = (description + " " + position).lower()

            # Check if skill is mentioned
            if search_skill in text:
                matching_projects.append(entry)

        # Calculate experience for matching projects
        result = calculate_total_experience(
            matching_projects, handle_overlaps=handle_overlaps
        )

        return {
            "skill": skill,
            "total_months": result.get("total_months", 0),
            "total_years": result.get("total_years", 0.0),
            "matching_projects": [
                {
                    "company": p.get("company"),
                    "position": p.get("position"),
                    "start": p.get("start"),
                    "end": p.get("end"),
                    "months": p.get("months"),
                }
                for p in result.get("periods", [])
            ],
            "projects_count": len(matching_projects),
            "error": result.get("error"),
        }

    except Exception as e:
        logger.error(f"Error calculating skill experience for '{skill}': {e}")
        return {
            "skill": skill,
            "total_months": None,
            "total_years": None,
            "matching_projects": [],
            "projects_count": 0,
            "error": str(e),
        }


def calculate_multiple_skills_experience(
    experience: List[Dict[str, Optional[str]]],
    skills: List[str],
    *,
    case_sensitive: bool = False,
    handle_overlaps: bool = True,
) -> Dict[str, Union[Dict, List[Dict]]]:
    """
    Calculate experience for multiple skills across all projects.

    This is a convenience function that calls calculate_skill_experience
    for each skill in the list and returns aggregated results.

    Args:
        experience: List of experience dictionaries
        skills: List of skill names to search for
        case_sensitive: Whether skill matching should be case-sensitive
        handle_overlaps: If True, merge overlapping periods for each skill

    Returns:
        Dictionary containing:
            - skills: Dictionary mapping skill name to experience data
            - summary: List of skills sorted by total experience (descending)
            - total_skills: Number of skills analyzed

    Examples:
        >>> experience = [
        ...     {
        ...         "start": "2020-05-01",
        ...         "end": "2023-02-01",
        ...         "description": "Java and Python development"
        ...     },
        ... ]
        >>> result = calculate_multiple_skills_experience(experience, ["Java", "Python"])
        >>> result["summary"][0]["skill"]  # Most experienced skill
        'Java'
    """
    if not skills:
        return {
            "skills": {},
            "summary": [],
            "total_skills": 0,
        }

    skills_data = {}

    for skill in skills:
        result = calculate_skill_experience(
            experience,
            skill,
            case_sensitive=case_sensitive,
            handle_overlaps=handle_overlaps,
        )
        skills_data[skill] = result

    # Create summary sorted by experience
    summary = sorted(
        [
            {
                "skill": skill,
                "total_months": data.get("total_months", 0),
                "total_years": data.get("total_years", 0.0),
                "projects_count": data.get("projects_count", 0),
            }
            for skill, data in skills_data.items()
        ],
        key=lambda x: x["total_months"] or 0,
        reverse=True,
    )

    return {
        "skills": skills_data,
        "summary": summary,
        "total_skills": len(skills),
    }


def format_experience_summary(
    experience_data: Dict[str, Optional[Union[int, str, List]]],
    *,
    include_periods: bool = False,
) -> str:
    """
    Format experience calculation results as a human-readable summary.

    Args:
        experience_data: Result dictionary from calculate_total_experience or
            calculate_skill_experience
        include_periods: Whether to include detailed period information

    Returns:
        Formatted string with experience summary

    Examples:
        >>> result = calculate_total_experience(experience)
        >>> print(format_experience_summary(result))
        Total Experience: 3 years and 10 months (46 months)
    """
    if experience_data.get("error"):
        return f"Error: {experience_data['error']}"

    total_months = experience_data.get("total_months")
    total_years = experience_data.get("total_years")

    if total_months is None:
        return "Unable to calculate experience"

    # Convert to years and months
    years = int(total_months // 12)
    months = int(total_months % 12)

    parts = []
    if years > 0:
        parts.append(f"{years} year{'s' if years != 1 else ''}")
    if months > 0 or years == 0:
        parts.append(f"{months} month{'s' if months != 1 else ''}")

    summary = "Total Experience: " + " and ".join(parts)
    summary += f" ({total_months} months)"

    if include_periods:
        periods = experience_data.get("periods", [])
        if periods:
            summary += "\n\nWork Periods:"
            for period in periods:
                company = period.get("company") or "Unknown"
                position = period.get("position") or "Unknown"
                months = period.get("months", 0)
                start = period.get("start", "Unknown")
                end = period.get("end") or "Present"
                summary += (
                    f"\n  - {company}: {position} ({start} to {end}) - {months} months"
                )

    return summary


def extract_experience_section(text: str) -> str:
    """
    Извлечь только секцию Experience из текста резюме.

    Эта функция находит секцию опыта работы и игнорирует образование,
    чтобы избежать учёта периода обучения как рабочего опыта.

    Args:
        text: Полный текст резюме

    Returns:
        Текст только секции Experience или пустую строку, если секция не найдена

    Examples:
        >>> text = "Skills: Python, Java\\nExperience\\nWork at Company A\\nEducation\\nUniversity"
        >>> result = extract_experience_section(text)
        >>> "Experience" in result
        True
        >>> "Education" in result
        False
    """
    if not text:
        return ""

    # Привести к нижнему регистру для поиска
    text_lower = text.lower()

    # Паттерны для начала секции опыта (искать как отдельные слова)
    # Используем регулярные выражения для поиска заголовков секций
    import re

    # Заголовки секций опыта (как отдельные строки или с двоеточием)
    experience_patterns = [
        r'\nexperience\s*\n',           # "Experience" на отдельной строке
        r'\nexperience:',                # "Experience:" с двоеточием
        r'\nwork experience\s*\n',       # "Work Experience"
        r'\nwork experience:',           # "Work Experience:"
        r'\nwork history\s*\n',         # "Work History"
        r'\nemployment\s*\n',           # "Employment"
        r'\nprofessional experience\s*\n',  # "Professional Experience"
        r'\nопыт работы\s*\n',          # "опыт работы" (на отдельной строке)
        r'\nопыт:',                     # "опыт:"
    ]

    # Паттерны для окончания секции опыта (начало других секций)
    section_end_patterns = [
        r'\neducation\s*\n',            # "Education" на отдельной строке
        r'\neducation:',                # "Education:"
        r'\nобразование\s*\n',         # "образование"
        r'\nskills\s*\n',              # "Skills"
        r'\nнавыки\s*\n',              # "навыки"
        r'\nlanguages\s*\n',           # "Languages"
        r'\nязыки\s*\n',               # "языки"
        r'\nprojects\s*\n',            # "Projects"
        r'\nпроекты\s*\n',            # "проекты"
        r'\ncertifications\s*\n',      # "Certifications"
        r'\nсертификации\s*\n',       # "сертификации"
        r'\nrecommendations\s*\n',     # "Recommendations"
        r'\nрекомендации\s*\n',       # "рекомендации"
    ]

    # Найти начало секции Experience
    experience_start = len(text)  # По умолчанию до конца
    experience_match = None

    for pattern in experience_patterns:
        match = re.search(pattern, text_lower)
        if match:
            start_pos = match.start()
            # Находим позицию в оригинальном тексте
            if start_pos < experience_start:
                experience_start = start_pos
                experience_match = match

    # Если секция Experience не найдена, вернуть весь текст
    if experience_match is None:
        logger.debug("No Experience section found, using full text")
        # Проверить, есть ли вообще секции в резюме
        has_sections = any(re.search(p, text_lower) for p in section_end_patterns)
        if not has_sections:
            return text
        return ""

    # Найти конец секции Experience (начало следующей секции)
    experience_end = len(text)

    # Ищем конец только ПОСЛЕ начала секции Experience
    for pattern in section_end_patterns:
        # Ищем совпадение после начала секции Experience
        remaining_text = text_lower[experience_start:]
        match = re.search(pattern, remaining_text)
        if match:
            end_pos = experience_start + match.start()
            if end_pos < experience_end:
                experience_end = end_pos

    # Извлечь секцию Experience
    experience_text = text[experience_start:experience_end].strip()

    logger.debug(f"Extracted experience section: {len(experience_text)} characters")
    return experience_text


def extract_dates_from_text(text: str, experience_only: bool = True) -> List[str]:
    """
    Извлечь все даты из текста резюме с использованием регулярных выражений.

    Поддерживает множество форматов дат:
    - Jun 2025, June 2025
    - 06 2025, 06/2025, 06.2025, 06-2025
    - 2025-06, 2025/06, 2025.06
    - 2025-06-15, 15.06.2025

    Args:
        text: Текст резюме для поиска дат
        experience_only: Если True, извлекать даты только из секции Experience

    Returns:
        Список найденных строк дат

    Examples:
        >>> text = "Worked from Jun 2020 to Dec 2023, then 01 2024 to present"
        >>> dates = extract_dates_from_text(text)
        >>> len(dates) >= 4
        True
    """
    if not text:
        return []

    # Если нужно только из секции Experience, извлечь её
    if experience_only:
        text = extract_experience_section(text)
        if not text:
            logger.debug("No experience section found in resume")
            return []

    # Месяцы на английском (краткие и полные)
    months_en = r"(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec|January|February|March|April|May|June|July|August|September|October|November|December)"

    # Месяцы на русском (краткие и полные)
    months_ru = r"(?:янв|фев|мар|апр|май|июн|июл|авг|сен|окт|ноя|дек|января|февраля|марта|апреля|мая|июня|июля|августа|сентября|октября|ноября|декабря)"

    # Паттерны для поиска дат - в порядке приоритета (от более конкретных к общим)
    patterns = [
        # Число.месяц.год: 15.06.2025, 15.06.25 (самый конкретный)
        r"\b(\d{1,2})\.(\d{1,2})\.(\d{2,4})\b",
        # Год-месяц-день: 2025-06-15
        r"\b(\d{4})-(\d{1,2})-(\d{1,2})\b",
        # Месяц/год: 06/2025, 06-2025
        r"\b(\d{1,2})[/-](\d{4})\b",
        # Год-месяц: 2025-06, 2025/06, 2025.06
        r"\b(\d{4})[/.-](\d{1,2})\b",
        # Число месяц год: 15 Jun 2025, 06 December 2023
        rf"\b(\d{{1,2}})\s+(?:{months_en})\s+(\d{{4}})\b",
    ]

    dates_found = []

    # Сначала ищем даты без разделителей (формат ГГГГ-ГГГГ, например 2020-2021 без тире)
    # В резюме это выглядит как 20202021 (8 цифр подряд)
    date_ranges_without_separators = re.findall(r"\b(\d{4})(\d{4})\b", text)
    logger.debug(f"Found {len(date_ranges_without_separators)} 8-digit date patterns")
    for year1, year2 in date_ranges_without_separators:
        # Проверяем, что вторая часть тоже валидный год (01-99)
        try:
            year2_int = int(year2)
            if 0 <= year2_int <= 99:  # Суффикс года от 00 до 99
                # Формируем два полных года
                full_year1 = f"20{year1}" if len(year1) == 2 else year1
                full_year2 = f"20{year2}" if len(year2) == 2 else year2
                # Добавляем как январь (по умолчанию, так как месяц неизвестен)
                dates_found.append(f"01 {full_year1}")  # Начало периода (январь первого года)
                dates_found.append(f"01 {full_year2}")  # Начало периода (январь второго года)
                logger.debug(f"Extracted years from {year1}-{year2}: 01/{full_year1} and 01/{full_year2}")
        except ValueError as e:
            logger.debug(f"ValueError parsing years: {e}")
            pass

    for pattern in patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        for match in matches:
            if isinstance(match, tuple):
                if len(match) == 3 and all(m for m in match):
                    # День.месяц.год или год-месяц-день
                    if len(match[2]) == 4:  # Год в конце: 15.06.2025
                        date_str = f"{match[1]}.{match[0]}.{match[2]}"
                    else:  # Год в начале: 2025-06-15
                        date_str = f"{match[0]}-{match[1]}-{match[2]}"
                    dates_found.append(date_str)
                elif len(match) == 2:
                    # Месяц/год, год/месяц, число/год и т.д.
                    if len(match[1]) == 4:  # Второе - год
                        if match[0].isdigit() and len(match[0]) <= 2:
                            date_str = f"{match[0]} {match[1]}"  # 06 2025
                        else:
                            continue  # Пропускаем нечисловые значения
                    else:  # Первое - год
                        if match[1].isdigit() and len(match[1]) <= 2:
                            date_str = f"{match[0]}-{match[1]}"  # 2025-06
                        else:
                            continue
                    dates_found.append(date_str)
            elif isinstance(match, str):
                # Для матчинга "Месяц Год" (Jun 2025, June 2025)
                month_map = {
                    'jan': '01', 'january': '01', 'feb': '02', 'february': '02',
                    'mar': '03', 'march': '03', 'apr': '04', 'april': '04',
                    'may': '05', 'jun': '06', 'june': '06', 'jul': '07', 'july': '07',
                    'aug': '08', 'august': '08', 'sep': '09', 'september': '09',
                    'oct': '10', 'october': '10', 'nov': '11', 'november': '11',
                    'dec': '12', 'december': '12'
                }
                # Извлекаем год из паттерна
                year_match = re.search(r'\b(\d{4})\b', match)
                if year_match:
                    year = year_match.group(1)
                    month_name = match.replace(year, '').strip().lower()
                    month_num = month_map.get(month_name)
                    if month_num:
                        date_str = f"{month_num} {year}"
                        dates_found.append(date_str)

    # Очистка и уникализация
    unique_dates = []
    seen = set()
    for date in dates_found:
        normalized = date.strip()
        if normalized and normalized not in seen:
            seen.add(normalized)
            unique_dates.append(normalized)

    return unique_dates


def calculate_total_experience_from_text(
    resume_text: str,
    *,
    skip_invalid: bool = True,
) -> Dict[str, Optional[Union[int, str, List]]]:
    """
    Рассчитать общий опыт работы из сырого текста резюме.

    Эта функция извлекает все даты из текста резюме, находит самую раннюю
    и самую позднюю дату, и рассчитывает общий опыт в месяцах и годах.

    Args:
        resume_text: Полный текст резюме
        skip_invalid: Если True, пропускать даты, которые не удаётся распарсить

    Returns:
        Словарь содержащий:
            - total_months: Общий опыт в месяцах
            - total_years: Общий опыт в годах (float)
            - earliest_date: Самая ранняя найденная дата
            - latest_date: Самая поздняя найденная дата
            - all_dates: Все найденные даты
            - date_count: Количество найденных дат
            - error: Сообщение об ошибке, если расчёт не удался

    Examples:
        >>> text = "Experience: Jun 2020 - Dec 2023 at Company A, then Jan 2024 - present at Company B"
        >>> result = calculate_total_experience_from_text(text)
        >>> result["total_months"]
        55  # Примерно 4 года 7 месяцев
        >>> result["total_years"]
        4.58
    """
    if not resume_text:
        return {
            "total_months": 0,
            "total_years": 0.0,
            "earliest_date": None,
            "latest_date": None,
            "all_dates": [],
            "date_count": 0,
            "error": "Resume text is empty",
        }

    try:
        # Извлечь даты только из секции Experience
        # Это исключает даты из секции Education
        date_strings = extract_dates_from_text(resume_text, experience_only=True)

        if not date_strings:
            return {
                "total_months": 0,
                "total_years": 0.0,
                "earliest_date": None,
                "latest_date": None,
                "all_dates": [],
                "date_count": 0,
                "error": "No dates found in resume",
            }

        # Распарсить даты
        parsed_dates = []
        invalid_dates = []

        for date_str in date_strings:
            try:
                parsed = _parse_date(date_str)
                if parsed:
                    parsed_dates.append((date_str, parsed))
            except ValueError as e:
                invalid_dates.append(date_str)
                if not skip_invalid:
                    raise ValueError(f"Invalid date '{date_str}': {e}")

        if not parsed_dates:
            return {
                "total_months": 0,
                "total_years": 0.0,
                "earliest_date": None,
                "latest_date": None,
                "all_dates": date_strings,
                "date_count": len(date_strings),
                "error": "No valid dates could be parsed",
            }

        # Найти самую раннюю и самую позднюю дату
        parsed_dates.sort(key=lambda x: x[1])  # Сортировка по datetime
        earliest = parsed_dates[0]
        latest = parsed_dates[-1]

        # Рассчитать общий опыт
        total_months = _calculate_months_between(earliest[1], latest[1])
        total_years = round(total_months / 12, 2)

        return {
            "total_months": total_months,
            "total_years": total_years,
            "earliest_date": earliest[0],  # Оригинальная строка
            "latest_date": latest[0],     # Оригинальная строка
            "earliest_parsed": earliest[1].isoformat(),
            "latest_parsed": latest[1].isoformat(),
            "all_dates": date_strings,
            "date_count": len(date_strings),
            "invalid_dates": invalid_dates if invalid_dates else None,
            "invalid_count": len(invalid_dates),
        }

    except Exception as e:
        logger.error(f"Error calculating experience from text: {e}", exc_info=True)
        return {
            "total_months": None,
            "total_years": None,
            "earliest_date": None,
            "latest_date": None,
            "all_dates": [],
            "date_count": 0,
            "error": str(e),
        }


def format_experience_from_text(
    experience_data: Dict[str, Optional[Union[int, str, List]]],
    language: str = "ru",
) -> str:
    """
    Отформатировать результаты расчёта опыта из текста как читаемую сводку.

    Args:
        experience_data: Результат словарь из calculate_total_experience_from_text
        language: Язык форматирования ('ru' или 'en')

    Returns:
        Отформатированная строка со сводкой опыта

    Examples:
        >>> result = calculate_total_experience_from_text(resume_text)
        >>> print(format_experience_from_text(result))
        Общий опыт: 4 года 7 месяцев (55 месяцев)
    """
    if experience_data.get("error"):
        return f"Ошибка: {experience_data['error']}"

    total_months = experience_data.get("total_months")
    total_years = experience_data.get("total_years")

    if total_months is None:
        return "Не удалось рассчитать опыт"

    # Конвертировать в годы и месяцы
    years = int(total_months // 12)
    months = int(total_months % 12)

    if language == "ru":
        # Русское склонение
        if years == 1:
            years_str = "1 год"
        elif 2 <= years % 10 <= 4:
            years_str = f"{years} года"
        else:
            years_str = f"{years} лет"

        if months == 1:
            months_str = "1 месяц"
        elif 2 <= months % 10 <= 4:
            months_str = f"{months} месяца"
        else:
            months_str = f"{months} месяцев"

        parts = []
        if years > 0:
            parts.append(years_str)
        if months > 0 or years == 0:
            parts.append(months_str)

        summary = "Общий опыт: " + " и ".join(parts)
        summary += f" ({total_months} месяцев)"

        if experience_data.get("earliest_date") and experience_data.get("latest_date"):
            summary += f"\nПериод: с {experience_data['earliest_date']} по {experience_data['latest_date']}"

    else:
        # English formatting
        parts = []
        if years > 0:
            parts.append(f"{years} year{'s' if years != 1 else ''}")
        if months > 0 or years == 0:
            parts.append(f"{months} month{'s' if months != 1 else ''}")

        summary = "Total experience: " + " and ".join(parts)
        summary += f" ({total_months} months)"

        if experience_data.get("earliest_date") and experience_data.get("latest_date"):
            summary += f"\nPeriod: from {experience_data['earliest_date']} to {experience_data['latest_date']}"

    return summary
