import React, { useState } from 'react';
import { Typography, Box, Tabs, Tab, Container } from '@mui/material';
import { useParams } from 'react-router-dom';
import { useTranslation } from 'react-i18next';
import AnalysisResults from '@components/AnalysisResults';
import VacancyMatchResults from '@components/VacancyMatchResults';

/**
 * Компонент страницы результатов
 *
 * Отображает полные результаты анализа резюме, включая:
 * - Обнаружение ошибок с значками критичности
 * - Проверку грамматики и орфографии
 * - Извлечение ключевых слов и навыков
 * - Сводку об опыте
 * - Подсветку навыков (зеленый - совпавшие, красный - отсутствующие)
 */
const ResultsPage: React.FC = () => {
  const { id } = useParams<{ id: string }>();
  const { t } = useTranslation();
  const [activeTab, setActiveTab] = useState(0);

  if (!id) {
    return (
      <Box>
        <Typography variant="h4" component="h1" gutterBottom fontWeight={600}>
          {t('results.title')}
        </Typography>
        <Typography variant="body1" color="error.main">
          {t('results.noResumeId')}
        </Typography>
      </Box>
    );
  }

  const handleTabChange = (_event: React.SyntheticEvent, newValue: number) => {
    setActiveTab(newValue);
  };

  return (
    <Container maxWidth="lg">
      <Box sx={{ mb: 3 }}>
        <Typography variant="h4" component="h1" gutterBottom fontWeight={600}>
          {t('results.title')}
        </Typography>
        <Typography variant="body2" color="text.secondary">
          Resume ID: {id}
        </Typography>
      </Box>

      <Box sx={{ borderBottom: 1, borderColor: 'divider', mb: 3 }}>
        <Tabs value={activeTab} onChange={handleTabChange} aria-label="results tabs">
          <Tab label="Analysis" />
          <Tab label="Vacancy Matches" />
        </Tabs>
      </Box>

      {activeTab === 0 && <AnalysisResults resumeId={id} />}
      {activeTab === 1 && <VacancyMatchResults resumeId={id} />}
    </Container>
  );
};

export default ResultsPage;
