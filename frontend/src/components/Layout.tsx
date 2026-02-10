import React, { ReactNode, useState } from 'react';
import { Outlet, Link, useLocation } from 'react-router-dom';
import { useTranslation } from 'react-i18next';
import {
  AppBar,
  Toolbar,
  Typography,
  Container,
  Box,
  Button,
  Menu,
  MenuItem,
} from '@mui/material';
import {
  Description as ResumeIcon,
  Work as WorkIcon,
  Person as PersonIcon,
  BusinessCenter as RecruiterIcon,
  ExpandMore as ExpandMoreIcon,
} from '@mui/icons-material';
import LanguageSwitcher from './LanguageSwitcher';

/**
 * Компонент макета приложения
 *
 * Обеспечивает последовательную структуру приложения с панелью навигации и основной областью контента.
 * Использует Outlet React Router для отрисовки дочерних маршрутов.
 */
interface LayoutProps {
  children?: ReactNode;
}

const Layout: React.FC<LayoutProps> = ({ children }) => {
  const { t } = useTranslation();
  const location = useLocation();
  const [jobSeekerAnchorEl, setJobSeekerAnchorEl] = useState<null | HTMLElement>(null);
  const [recruiterAnchorEl, setRecruiterAnchorEl] = useState<null | HTMLElement>(null);

  const jobSeekerMenuOpen = Boolean(jobSeekerAnchorEl);
  const recruiterMenuOpen = Boolean(recruiterAnchorEl);

  // Пункты меню модуля соискателя
  const jobSeekerItems = [
    { labelKey: 'nav.browseJobs', path: '/jobs', icon: <WorkIcon fontSize="small" /> },
    { labelKey: 'nav.uploadResumeNav', path: '/jobs/upload', icon: <ResumeIcon fontSize="small" /> },
    { labelKey: 'nav.myApplications', path: '/jobs/applications', icon: <PersonIcon fontSize="small" /> },
  ];

  // Пункты меню модуля рекрутера
  const recruiterItems = [
    { labelKey: 'nav.dashboard', path: '/recruiter', icon: <RecruiterIcon fontSize="small" /> },
    { labelKey: 'nav.manageVacancies', path: '/recruiter/vacancies', icon: <WorkIcon fontSize="small" /> },
    { labelKey: 'nav.resumeDatabase', path: '/recruiter/resumes', icon: <PersonIcon fontSize="small" /> },
    { labelKey: 'nav.searchCandidates', path: '/recruiter/search', icon: <RecruiterIcon fontSize="small" /> },
  ];

  return (
    <Box sx={{ display: 'flex', flexDirection: 'column', minHeight: '100vh' }}>
      {/* Панель приложения / Навигация */}
      <AppBar position="static" elevation={2}>
        <Container maxWidth="lg">
          <Toolbar disableGutters>
            {/* Логотип / Бренд */}
            <Box sx={{ display: 'flex', alignItems: 'center', mr: 4 }}>
              <ResumeIcon sx={{ mr: 1, fontSize: 32 }} />
              <Typography
                variant="h6"
                component={Link}
                to="/"
                sx={{
                  fontWeight: 700,
                  color: 'inherit',
                  textDecoration: 'none',
                  letterSpacing: '-0.5px',
                }}
              >
                {t('appName')}
              </Typography>
            </Box>

            {/* Ссылки навигации - на основе модулей */}
            <Box sx={{ flexGrow: 1, display: 'flex', gap: 1 }}>
              {/* Модуль соискателя */}
              <Button
                color="inherit"
                startIcon={<WorkIcon />}
                endIcon={<ExpandMoreIcon />}
                onClick={(e) => setJobSeekerAnchorEl(e.currentTarget)}
                sx={{
                  textTransform: 'none',
                  fontWeight: 500,
                  borderRadius: 1,
                  px: 2,
                }}
              >
                {t('nav.findJobs')}
              </Button>
              <Menu
                anchorEl={jobSeekerAnchorEl}
                open={jobSeekerMenuOpen}
                onClose={() => setJobSeekerAnchorEl(null)}
                anchorOrigin={{
                  vertical: 'bottom',
                  horizontal: 'left',
                }}
                transformOrigin={{
                  vertical: 'top',
                  horizontal: 'left',
                }}
              >
                {jobSeekerItems.map((item) => (
                  <MenuItem
                    key={item.path}
                    component={Link}
                    to={item.path}
                    onClick={() => setJobSeekerAnchorEl(null)}
                    selected={location.pathname === item.path}
                  >
                    <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, minWidth: 180 }}>
                      {item.icon}
                      <Typography variant="body2">{t(item.labelKey)}</Typography>
                    </Box>
                  </MenuItem>
                ))}
              </Menu>

              {/* Модуль рекрутера */}
              <Button
                color="inherit"
                startIcon={<RecruiterIcon />}
                endIcon={<ExpandMoreIcon />}
                onClick={(e) => setRecruiterAnchorEl(e.currentTarget)}
                sx={{
                  textTransform: 'none',
                  fontWeight: 500,
                  borderRadius: 1,
                  px: 2,
                }}
              >
                {t('nav.findEmployees')}
              </Button>
              <Menu
                anchorEl={recruiterAnchorEl}
                open={recruiterMenuOpen}
                onClose={() => setRecruiterAnchorEl(null)}
                anchorOrigin={{
                  vertical: 'bottom',
                  horizontal: 'left',
                }}
                transformOrigin={{
                  vertical: 'top',
                  horizontal: 'left',
                }}
              >
                {recruiterItems.map((item) => (
                  <MenuItem
                    key={item.path}
                    component={Link}
                    to={item.path}
                    onClick={() => setRecruiterAnchorEl(null)}
                    selected={location.pathname === item.path}
                  >
                    <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, minWidth: 180 }}>
                      {item.icon}
                      <Typography variant="body2">{t(item.labelKey)}</Typography>
                    </Box>
                  </MenuItem>
                ))}
              </Menu>
            </Box>

            {/* Переключатель языка */}
            <LanguageSwitcher />
          </Toolbar>
        </Container>
      </AppBar>

      {/* Основная область контента */}
      {children ? (
        <Box sx={{ flexGrow: 1 }}>
          {children}
        </Box>
      ) : (
        <Box sx={{ flexGrow: 1, py: 4 }}>
          <Container maxWidth="lg">
            <Outlet />
          </Container>
        </Box>
      )}

      {/* Футер */}
      <Box
        component="footer"
        sx={{
          py: 3,
          px: 2,
          mt: 'auto',
          backgroundColor: (theme) =>
            theme.palette.mode === 'light'
              ? theme.palette.grey[200]
              : theme.palette.grey[800],
        }}
      >
        <Container maxWidth="lg">
          <Typography variant="body2" color="text.secondary" align="center">
            {t('footer.copyright', { year: new Date().getFullYear() })}
          </Typography>
        </Container>
      </Box>
    </Box>
  );
};

export default Layout;
