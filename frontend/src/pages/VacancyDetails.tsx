import React, { useState, useEffect } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import {
  Box,
  Container,
  Paper,
  Typography,
  Button,
  Chip,
  Stack,
  Divider,
  Grid,
  Card,
  CardContent,
  CircularProgress,
  Alert,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  List,
  ListItem,
  ListItemText,
  ListItemAvatar,
  Avatar,
  IconButton,
  TextField,
  InputAdornment,
} from '@mui/material';
import {
  ArrowBack as ArrowBackIcon,
  Edit as EditIcon,
  Delete as DeleteIcon,
  Work as WorkIcon,
  Business as BusinessIcon,
  LocationOn as LocationIcon,
  Money as MoneyIcon,
  Person as PersonIcon,
  Search as SearchIcon,
  Compare as CompareIcon,
} from '@mui/icons-material';
import { useTranslation } from 'react-i18next';
import axios from 'axios';

interface Vacancy {
  id: string;
  title: string;
  description: string;
  location: string;
  work_format: string;
  required_skills: string[];
  min_experience_months: number;
  salary_min?: number;
  salary_max?: number;
  created_at: string;
}

interface Resume {
  id: string;
  filename: string;
  uploaded_at: string;
  raw_text?: string;
}

const VacancyDetails: React.FC = () => {
  const { id } = useParams<{ id: string }>();
  const navigate = useNavigate();
  const { t } = useTranslation();
  const [vacancy, setVacancy] = useState<Vacancy | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  // Resume matching dialog state
  const [matchDialogOpen, setMatchDialogOpen] = useState(false);
  const [resumes, setResumes] = useState<Resume[]>([]);
  const [resumesLoading, setResumesLoading] = useState(false);
  const [resumesError, setResumesError] = useState<string | null>(null);
  const [searchQuery, setSearchQuery] = useState('');

  useEffect(() => {
    const fetchVacancy = async () => {
      try {
        setLoading(true);
        const response = await axios.get(`/api/vacancies/${id}`);
        setVacancy(response.data);
      } catch (err) {
        setError(err instanceof Error ? err.message : t('vacancyDetails.failedToLoad'));
      } finally {
        setLoading(false);
      }
    };

    if (id) {
      fetchVacancy();
    }
  }, [id, t]);

  const handleDelete = async () => {
    if (!confirm(t('vacancyDetails.deleteConfirm'))) return;

    try {
      await axios.delete(`/api/vacancies/${id}`);
      navigate('/jobs');
    } catch (err) {
      setError(err instanceof Error ? err.message : t('vacancyDetails.failedToDelete'));
    }
  };

  const fetchResumes = async () => {
    setResumesLoading(true);
    setResumesError(null);

    try {
      const response = await axios.get('/api/resumes/?limit=100');
      setResumes(response.data);
    } catch (err) {
      setResumesError(err instanceof Error ? err.message : 'Не удалось загрузить резюме');
    } finally {
      setResumesLoading(false);
    }
  };

  const handleMatchDialogOpen = () => {
    setMatchDialogOpen(true);
    fetchResumes();
  };

  const handleMatchDialogClose = () => {
    setMatchDialogOpen(false);
    setSearchQuery('');
  };

  const handleResumeSelect = (resumeId: string) => {
    handleMatchDialogClose();
    navigate(`/compare/${resumeId}/${id}`);
  };

  const filteredResumes = resumes.filter((resume) =>
    resume.filename.toLowerCase().includes(searchQuery.toLowerCase())
  );

  if (loading) {
    return (
      <Box sx={{ display: 'flex', justifyContent: 'center', alignItems: 'center', minHeight: '50vh' }}>
        <CircularProgress />
      </Box>
    );
  }

  if (error || !vacancy) {
    return (
      <Container maxWidth="lg" sx={{ py: 4 }}>
        <Alert severity="error">{error || t('vacancyDetails.notFound')}</Alert>
      </Container>
    );
  }

  return (
    <Container maxWidth="lg" sx={{ py: 4 }}>
      {/* Заголовок */}
      <Box sx={{ mb: 3 }}>
        <Button
          startIcon={<ArrowBackIcon />}
          onClick={() => navigate('/jobs')}
          sx={{ mb: 2 }}
        >
          {t('common.back')}
        </Button>
      </Box>

      <Paper sx={{ p: 4 }}>
        {/* Название и действия */}
        <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', mb: 3 }}>
          <Box sx={{ flex: 1 }}>
            <Typography variant="h4" fontWeight={600} gutterBottom>
              {vacancy.title}
            </Typography>
            <Stack direction="row" spacing={1} mt={1}>
              <Chip
                icon={<LocationIcon />}
                label={vacancy.location || 'Remote'}
                size="small"
                variant="outlined"
              />
              <Chip
                label={vacancy.work_format || 'Full-time'}
                size="small"
                color="primary"
                variant="outlined"
              />
            </Stack>
          </Box>
          <Stack direction="row" spacing={1}>
            <Button
              startIcon={<CompareIcon />}
              variant="contained"
              color="primary"
              onClick={handleMatchDialogOpen}
            >
              {t('vacancyDetails.matchWithResume')}
            </Button>
            <Button
              startIcon={<EditIcon />}
              variant="outlined"
              onClick={() => navigate(`/jobs/${vacancy.id}/edit`)}
            >
              {t('vacancyDetails.editVacancy')}
            </Button>
            <Button
              startIcon={<DeleteIcon />}
              variant="outlined"
              color="error"
              onClick={handleDelete}
            >
              {t('vacancyDetails.deleteVacancy')}
            </Button>
          </Stack>
        </Box>

        <Divider sx={{ mb: 3 }} />

        <Grid container spacing={3}>
          {/* Детали */}
          <Grid item xs={12} md={8}>
            <Box sx={{ mb: 3 }}>
              <Typography variant="subtitle1" fontWeight={600} gutterBottom>
                {t('vacancyDetails.description')}
              </Typography>
              <Typography variant="body1" color="text.secondary" sx={{ whiteSpace: 'pre-wrap' }}>
                {vacancy.description || t('vacancyDetails.noDescription')}
              </Typography>
            </Box>
          </Grid>

          {/* Боковая панель */}
          <Grid item xs={12} md={4}>
            <Card variant="outlined">
              <CardContent>
                <Typography variant="subtitle2" fontWeight={600} gutterBottom>
                  {t('vacancyDetails.details')}
                </Typography>
                <Stack spacing={2}>
                  {vacancy.min_experience_months && (
                    <Box>
                      <Typography variant="caption" color="text.secondary">
                        {t('vacancyDetails.requiredExperience')}
                      </Typography>
                      <Typography variant="body1">
                        {Math.floor(vacancy.min_experience_months / 12)}+ {t('vacancyDetails.years')}
                      </Typography>
                    </Box>
                  )}
                  {(vacancy.salary_min || vacancy.salary_max) && (
                    <Box>
                      <Typography variant="caption" color="text.secondary">
                        {t('vacancyDetails.salary')}
                      </Typography>
                      <Typography variant="body1">
                        {vacancy.salary_min && vacancy.salary_max
                          ? `$${vacancy.salary_min} - $${vacancy.salary_max}`
                          : vacancy.salary_min
                            ? `$${vacancy.salary_min}+`
                            : `До $${vacancy.salary_max}`}
                      </Typography>
                    </Box>
                  )}
                </Stack>
              </CardContent>
            </Card>
          </Grid>
        </Grid>

        {/* Обязательные навыки */}
        {vacancy.required_skills && vacancy.required_skills.length > 0 && (
          <>
            <Divider sx={{ my: 3 }} />
            <Box>
              <Typography variant="subtitle1" fontWeight={600} gutterBottom>
                {t('vacancyDetails.requiredSkills')}
              </Typography>
              <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 1 }}>
                {vacancy.required_skills.map((skill) => (
                  <Chip key={skill} label={skill} size="small" />
                ))}
              </Box>
            </Box>
          </>
        )}
      </Paper>

      {/* Resume Matching Dialog */}
      <Dialog
        open={matchDialogOpen}
        onClose={handleMatchDialogClose}
        maxWidth="md"
        fullWidth
        PaperProps={{
          sx: { borderRadius: 2 },
        }}
      >
        <DialogTitle>
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
            <CompareIcon color="primary" />
            <Typography variant="h6" component="span">
              {t('vacancyDetails.matchDialog.title')}
            </Typography>
          </Box>
        </DialogTitle>

        <DialogContent>
          {/* Search Field */}
          <TextField
            fullWidth
            placeholder={t('vacancyDetails.matchDialog.searchPlaceholder')}
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            InputProps={{
              startAdornment: (
                <InputAdornment position="start">
                  <SearchIcon />
                </InputAdornment>
              ),
            }}
            sx={{ mb: 2 }}
          />

          {/* Resumes List */}
          {resumesLoading ? (
            <Box sx={{ display: 'flex', justifyContent: 'center', py: 4 }}>
              <CircularProgress />
            </Box>
          ) : resumesError ? (
            <Alert severity="error">{resumesError}</Alert>
          ) : filteredResumes.length === 0 ? (
            <Box sx={{ textAlign: 'center', py: 4 }}>
              <WorkIcon sx={{ fontSize: 48, color: 'text.secondary', mb: 2 }} />
              <Typography variant="body1" color="text.secondary">
                {searchQuery ? t('vacancyDetails.matchDialog.noResumesFound') : t('vacancyDetails.matchDialog.noResumes')}
              </Typography>
            </Box>
          ) : (
            <List sx={{ maxHeight: 400, overflow: 'auto' }}>
              {filteredResumes.map((resume) => (
                <ListItem
                  key={resume.id}
                  onClick={() => handleResumeSelect(resume.id)}
                  sx={{
                    borderRadius: 1,
                    mb: 1,
                    border: '1px solid',
                    borderColor: 'divider',
                    cursor: 'pointer',
                    '&:hover': {
                      bgcolor: 'action.hover',
                      borderColor: 'primary.main',
                    },
                  }}
                >
                  <ListItemAvatar>
                    <Avatar>
                      <PersonIcon />
                    </Avatar>
                  </ListItemAvatar>
                  <ListItemText
                    primary={resume.filename}
                    secondary={`${t('vacancyDetails.matchDialog.uploaded')}: ${new Date(resume.uploaded_at).toLocaleDateString('ru-RU')}`}
                  />
                  <IconButton edge="end" color="primary">
                    <CompareIcon />
                  </IconButton>
                </ListItem>
              ))}
            </List>
          )}
        </DialogContent>

        <DialogActions>
          <Button onClick={handleMatchDialogClose}>
            {t('vacancyDetails.matchDialog.cancel')}
          </Button>
        </DialogActions>
      </Dialog>
    </Container>
  );
};

export default VacancyDetails;
