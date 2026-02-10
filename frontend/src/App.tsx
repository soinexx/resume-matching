import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom';
import Layout from '@components/Layout';
import HomePage from '@pages/Home';
import UploadPage from '@pages/Upload';
import ResultsPage from '@pages/Results';
import ComparePage from '@pages/Compare';
import CompareVacancyPage from '@pages/CompareVacancy';
import AdminSynonymsPage from '@pages/AdminSynonyms';
import AdminAnalyticsPage from '@pages/AdminAnalytics';
import AnalyticsDashboardPage from '@pages/AnalyticsDashboard';
import VacancyListPage from '@pages/VacancyList';
import CreateVacancyPage from '@pages/CreateVacancy';
import VacancyDetailsPage from '@pages/VacancyDetails';
import ApplicationsPage from '@pages/Applications';
import ResumeDatabasePage from '@pages/ResumeDatabase';
import CandidateSearchPage from '@pages/CandidateSearch';
import RecruiterDashboardPage from '@pages/RecruiterDashboard';

/**
 * Главный компонент приложения
 *
 * Настраивает React Router со всеми маршрутами приложения.
 * Использует компонент Layout для обеспечения последовательной навигации и структуры.
 */
function App() {
  return (
    <BrowserRouter>
      <Routes>
        {/* Устаревшие маршруты - сохранены для совместимости */}
        <Route path="/upload" element={<Layout><UploadPage /></Layout>} />
        <Route path="/results/:id" element={<Layout><ResultsPage /></Layout>} />
        <Route path="/compare/:resumeId/:vacancyId" element={<Layout><ComparePage /></Layout>} />
        <Route path="/compare-vacancy/:vacancyId" element={<Layout><CompareVacancyPage /></Layout>} />

        {/* Маршруты модуля соискателя */}
        <Route path="/jobs/:id" element={<Layout><VacancyDetailsPage /></Layout>} />
        <Route path="/jobs" element={<Layout><VacancyListPage /></Layout>} />
        <Route path="/jobs/upload" element={<Layout><UploadPage /></Layout>} />
        <Route path="/jobs/results/:id" element={<Layout><ResultsPage /></Layout>} />
        <Route path="/jobs/applications" element={<Layout><ApplicationsPage /></Layout>} />

        {/* Маршруты модуля рекрутера */}
        <Route path="/recruiter" element={<Layout><RecruiterDashboardPage /></Layout>} />
        <Route path="/recruiter/vacancies" element={<Layout><VacancyListPage /></Layout>} />
        <Route path="/recruiter/vacancies/create" element={<Layout><CreateVacancyPage /></Layout>} />
        <Route path="/recruiter/vacancies/:id" element={<Layout><VacancyDetailsPage /></Layout>} />
        <Route path="/recruiter/vacancies/:id/edit" element={<Layout><CreateVacancyPage /></Layout>} />
        <Route path="/recruiter/resumes" element={<Layout><ResumeDatabasePage /></Layout>} />
        <Route path="/recruiter/search" element={<Layout><CandidateSearchPage /></Layout>} />
        <Route path="/recruiter/analytics" element={<Layout><AnalyticsDashboardPage /></Layout>} />

        {/* Отдельный маршрут для деталей вакансии (для прямого доступа) */}
        <Route path="/vacancies/:id" element={<Layout><VacancyDetailsPage /></Layout>} />

        {/* Страницы администратора */}
        <Route path="/admin" element={<Layout><Navigate to="/admin/synonyms" replace /></Layout>} />
        <Route path="/admin/synonyms" element={<Layout><AdminSynonymsPage /></Layout>} />
        <Route path="/admin/analytics" element={<Layout><AdminAnalyticsPage /></Layout>} />

        {/* Панель аналитики */}
        <Route path="/analytics" element={<Layout><AnalyticsDashboardPage /></Layout>} />

        {/* Главная страница - ДОЛЖНА БЫТЬ ПОСЛЕДНЕЙ! */}
        <Route path="/" element={<Layout><HomePage /></Layout>} />
      </Routes>
    </BrowserRouter>
  );
}

export default App;
