import React, { useState } from 'react';
import { ThemeProvider, createTheme } from '@mui/material/styles';
import CssBaseline from '@mui/material/CssBaseline';
import {
  AppBar,
  Toolbar,
  Typography,
  Button,
  Box,
  Avatar,
  Menu,
  MenuItem,
  Tabs,
  Tab
} from '@mui/material';
import FitnessCenterIcon from '@mui/icons-material/FitnessCenter';
import LogoutIcon from '@mui/icons-material/Logout';
import DashboardIcon from '@mui/icons-material/Dashboard';
import PlayArrowIcon from '@mui/icons-material/PlayArrow';
import BugReportIcon from '@mui/icons-material/BugReport';

import { AuthProvider, useAuth } from './contexts/AuthContext';
import Login from './components/Login';
import ExerciseSelector from './components/ExerciseSelector';
import ExerciseMonitor from './components/ExerciseMonitor';
import Dashboard from './components/Dashboard';
import MediaPipeDebug from './components/MediaPipeDebug';
import FloatingChatbot from './components/FloatingChatbot';
import Disclaimer from './components/Disclaimer';

// Create Material-UI theme with enhanced design
const theme = createTheme({
  palette: {
    mode: 'light',
    primary: {
      main: '#667eea',
      light: '#8fa5f7',
      dark: '#4959d1',
      contrastText: '#ffffff',
    },
    secondary: {
      main: '#764ba2',
      light: '#a477d1',
      dark: '#542575',
      contrastText: '#ffffff',
    },
    background: {
      default: '#f8fafc',
      paper: '#ffffff',
    },
    text: {
      primary: '#2d3748',
      secondary: '#4a5568',
    },
    error: {
      main: '#f56565',
    },
    warning: {
      main: '#ed8936',
    },
    success: {
      main: '#38a169',
    },
  },
  typography: {
    fontFamily: '"Inter", "Roboto", "Helvetica", "Arial", sans-serif',
    h1: {
      fontWeight: 700,
      fontSize: '2.5rem',
    },
    h2: {
      fontWeight: 700,
      fontSize: '2rem',
    },
    h3: {
      fontWeight: 600,
      fontSize: '1.75rem',
    },
    h4: {
      fontWeight: 600,
      fontSize: '1.5rem',
    },
    h5: {
      fontWeight: 600,
      fontSize: '1.25rem',
    },
    h6: {
      fontWeight: 600,
      fontSize: '1.125rem',
    },
    button: {
      textTransform: 'none',
      fontWeight: 500,
    },
  },
  shape: {
    borderRadius: 12,
  },
  shadows: [
    'none',
    '0px 2px 4px rgba(0,0,0,0.05)',
    '0px 4px 8px rgba(0,0,0,0.08)',
    '0px 6px 12px rgba(0,0,0,0.1)',
    '0px 8px 16px rgba(0,0,0,0.12)',
    '0px 12px 24px rgba(0,0,0,0.15)',
    '0px 16px 32px rgba(0,0,0,0.18)',
    '0px 20px 40px rgba(0,0,0,0.2)',
    '0px 24px 48px rgba(0,0,0,0.22)',
    '0px 28px 56px rgba(0,0,0,0.24)',
    '0px 32px 64px rgba(0,0,0,0.26)',
    '0px 36px 72px rgba(0,0,0,0.28)',
    '0px 40px 80px rgba(0,0,0,0.3)',
    '0px 44px 88px rgba(0,0,0,0.32)',
    '0px 48px 96px rgba(0,0,0,0.34)',
    '0px 52px 104px rgba(0,0,0,0.36)',
    '0px 56px 112px rgba(0,0,0,0.38)',
    '0px 60px 120px rgba(0,0,0,0.4)',
    '0px 64px 128px rgba(0,0,0,0.42)',
    '0px 68px 136px rgba(0,0,0,0.44)',
    '0px 72px 144px rgba(0,0,0,0.46)',
    '0px 76px 152px rgba(0,0,0,0.48)',
    '0px 80px 160px rgba(0,0,0,0.5)',
    '0px 84px 168px rgba(0,0,0,0.52)',
    '0px 88px 176px rgba(0,0,0,0.54)',
  ],
  components: {
    MuiButton: {
      styleOverrides: {
        root: {
          borderRadius: 10,
          textTransform: 'none',
          fontWeight: 500,
          boxShadow: 'none',
          '&:hover': {
            boxShadow: '0px 4px 12px rgba(102, 126, 234, 0.3)',
            transform: 'translateY(-1px)',
          },
          transition: 'all 0.2s ease-in-out',
        },
      },
    },
    MuiPaper: {
      styleOverrides: {
        root: {
          borderRadius: 12,
        },
      },
    },
    MuiCard: {
      styleOverrides: {
        root: {
          borderRadius: 16,
          boxShadow: '0px 4px 20px rgba(0,0,0,0.08)',
          border: '1px solid rgba(0,0,0,0.05)',
        },
      },
    },
    MuiAppBar: {
      styleOverrides: {
        root: {
          background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
          boxShadow: '0px 4px 20px rgba(102, 126, 234, 0.3)',
        },
      },
    },
    MuiTab: {
      styleOverrides: {
        root: {
          borderRadius: 8,
          margin: '0 4px',
          minHeight: 48,
          '&.Mui-selected': {
            backgroundColor: 'rgba(255,255,255,0.1)',
          },
        },
      },
    },
  },
});

type AppView = 'exercises' | 'monitor' | 'dashboard' | 'debug';

const AppContent: React.FC = () => {
  const { currentUser, logout } = useAuth();
  const [currentView, setCurrentView] = useState<AppView>('exercises');
  const [selectedExercise, setSelectedExercise] = useState<string>('');
  const [anchorEl, setAnchorEl] = useState<null | HTMLElement>(null);
  const [disclaimerAccepted, setDisclaimerAccepted] = useState<boolean>(false);

  // Auto-logout on app close/refresh and reset disclaimer
  React.useEffect(() => {
    const handleBeforeUnload = () => {
      // Clear all session data when app is closing
      if (currentUser) {
        const disclaimerKey = `disclaimer_accepted_${currentUser.uid}`;
        sessionStorage.removeItem(disclaimerKey);
        localStorage.removeItem('disclaimerAccepted'); // Clear any persistent storage
        // Note: logout() in beforeunload might not complete, but Firebase will handle session cleanup
      }
    };

    const handleVisibilityChange = () => {
      // Additional cleanup when tab becomes hidden
      if (document.visibilityState === 'hidden' && currentUser) {
        const disclaimerKey = `disclaimer_accepted_${currentUser.uid}`;
        sessionStorage.removeItem(disclaimerKey);
      }
    };

    // Add event listeners
    window.addEventListener('beforeunload', handleBeforeUnload);
    document.addEventListener('visibilitychange', handleVisibilityChange);

    // Cleanup function
    return () => {
      window.removeEventListener('beforeunload', handleBeforeUnload);
      document.removeEventListener('visibilitychange', handleVisibilityChange);
    };
  }, [currentUser]);

  // Check disclaimer acceptance on component mount - always start fresh
  React.useEffect(() => {
    if (currentUser) {
      console.log('User logged in, resetting disclaimer state');
      // Always start with disclaimer not accepted on app open
      setDisclaimerAccepted(false);
      
      // Clear any existing disclaimer data to ensure fresh start
      const disclaimerKey = `disclaimer_accepted_${currentUser.uid}`;
      sessionStorage.removeItem(disclaimerKey);
      localStorage.removeItem('disclaimerAccepted');
    }
  }, [currentUser]);

  // Clear disclaimer acceptance on logout
  const handleLogout = async () => {
    try {
      if (currentUser) {
        const disclaimerKey = `disclaimer_accepted_${currentUser.uid}`;
        sessionStorage.removeItem(disclaimerKey);
        // Reset disclaimer state before logout to ensure it shows after login
        setDisclaimerAccepted(false);
      }
      await logout();
      handleClose();
    } catch (error) {
      console.error('Logout error:', error);
    }
  };

  const handleDisclaimerAccept = () => {
    console.log('Disclaimer accepted by user');
    if (currentUser) {
      // Only store in session storage for current session
      const disclaimerKey = `disclaimer_accepted_${currentUser.uid}`;
      sessionStorage.setItem(disclaimerKey, 'true');
      setDisclaimerAccepted(true);
      
      // Add a timestamp to track when disclaimer was accepted
      const timestamp = new Date().toISOString();
      sessionStorage.setItem(`disclaimer_timestamp_${currentUser.uid}`, timestamp);
    }
  };

  const handleMenu = (event: React.MouseEvent<HTMLElement>) => {
    setAnchorEl(event.currentTarget);
  };

  const handleClose = () => {
    setAnchorEl(null);
  };

  const handleExerciseSelect = (exercise: string) => {
    setSelectedExercise(exercise);
    setCurrentView('monitor');
  };

  const handleBackToExercises = () => {
    setCurrentView('exercises');
    setSelectedExercise('');
  };

  const handleTabChange = (event: React.SyntheticEvent, newValue: AppView) => {
    setCurrentView(newValue);
  };

  console.log('Current user:', currentUser);
  console.log('Disclaimer accepted:', disclaimerAccepted);

  if (!currentUser) {
    console.log('No user - showing login');
    return <Login />;
  }

  // Show disclaimer if not accepted yet
  if (!disclaimerAccepted) {
    console.log('User logged in but disclaimer not accepted - showing disclaimer');
    return <Disclaimer onAccept={handleDisclaimerAccept} />;
  }

  console.log('User logged in and disclaimer accepted - showing main app');

  return (
    <Box sx={{ 
      flexGrow: 1, 
      minHeight: '100vh',
      background: 'linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%)',
    }}>
      {/* App Bar */}
      <AppBar 
        position="static" 
        elevation={0}
        sx={{
          background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
          backdropFilter: 'blur(10px)',
          borderBottom: '1px solid rgba(255,255,255,0.1)',
        }}
      >
        <Toolbar sx={{ py: 1 }}>
          <Box sx={{ display: 'flex', alignItems: 'center', mr: 3 }}>
            <FitnessCenterIcon sx={{ 
              mr: 1, 
              fontSize: 32,
              filter: 'drop-shadow(0px 2px 4px rgba(0,0,0,0.2))',
            }} />
            <Typography 
              variant="h6" 
              component="div" 
              sx={{ 
                fontWeight: 700,
                background: 'linear-gradient(45deg, #ffffff 30%, #f0f8ff 90%)',
                backgroundClip: 'text',
                WebkitBackgroundClip: 'text',
                WebkitTextFillColor: 'transparent',
                textShadow: '0px 2px 4px rgba(0,0,0,0.1)',
              }}
            >
              POSECRAFT
            </Typography>
          </Box>
          
          {/* Navigation Tabs */}
          <Box sx={{ flexGrow: 1, display: 'flex', justifyContent: 'center' }}>
            <Tabs 
              value={currentView} 
              onChange={handleTabChange}
              textColor="inherit"
              indicatorColor="secondary"
              sx={{
                '& .MuiTabs-indicator': {
                  backgroundColor: '#ffffff',
                  height: 3,
                  borderRadius: 2,
                },
              }}
            >
              <Tab 
                icon={<PlayArrowIcon />} 
                label="Exercises" 
                value="exercises"
                sx={{ 
                  color: 'rgba(255,255,255,0.8)',
                  fontWeight: 500,
                  '&.Mui-selected': {
                    color: '#ffffff',
                    fontWeight: 600,
                  },
                  '&:hover': {
                    backgroundColor: 'rgba(255,255,255,0.1)',
                  },
                }}
              />
              <Tab 
                icon={<DashboardIcon />} 
                label="Dashboard" 
                value="dashboard"
                sx={{ 
                  color: 'rgba(255,255,255,0.8)',
                  fontWeight: 500,
                  '&.Mui-selected': {
                    color: '#ffffff',
                    fontWeight: 600,
                  },
                  '&:hover': {
                    backgroundColor: 'rgba(255,255,255,0.1)',
                  },
                }}
              />
              <Tab 
                icon={<BugReportIcon />} 
                label="Debug" 
                value="debug"
                sx={{ 
                  color: 'rgba(255,255,255,0.8)',
                  fontWeight: 500,
                  '&.Mui-selected': {
                    color: '#ffffff',
                    fontWeight: 600,
                  },
                  '&:hover': {
                    backgroundColor: 'rgba(255,255,255,0.1)',
                  },
                }}
              />
            </Tabs>
          </Box>

          {/* User Menu */}
          <Box sx={{ display: 'flex', alignItems: 'center' }}>
            <Typography 
              variant="body2" 
              sx={{ 
                mr: 2,
                color: 'rgba(255,255,255,0.9)',
                fontWeight: 500,
              }}
            >
              {currentUser.displayName || currentUser.email}
            </Typography>
            <Button
              onClick={handleMenu}
              sx={{ 
                p: 0,
                borderRadius: '50%',
                '&:hover': {
                  transform: 'scale(1.05)',
                  boxShadow: '0px 4px 12px rgba(0,0,0,0.2)',
                },
                transition: 'all 0.2s ease',
              }}
            >
              <Avatar 
                src={currentUser.photoURL || undefined}
                sx={{ 
                  width: 40, 
                  height: 40,
                  border: '2px solid rgba(255,255,255,0.3)',
                  boxShadow: '0px 4px 12px rgba(0,0,0,0.2)',
                }}
              >
                {currentUser.displayName?.[0] || currentUser.email?.[0]}
              </Avatar>
            </Button>
            <Menu
              anchorEl={anchorEl}
              open={Boolean(anchorEl)}
              onClose={handleClose}
              transformOrigin={{ horizontal: 'right', vertical: 'top' }}
              anchorOrigin={{ horizontal: 'right', vertical: 'bottom' }}
              PaperProps={{
                sx: {
                  mt: 1,
                  borderRadius: 2,
                  boxShadow: '0px 8px 24px rgba(0,0,0,0.15)',
                  border: '1px solid rgba(0,0,0,0.05)',
                },
              }}
            >
              <MenuItem 
                onClick={handleLogout}
                sx={{
                  py: 1.5,
                  px: 2,
                  '&:hover': {
                    backgroundColor: 'rgba(244, 67, 54, 0.08)',
                    color: '#f44336',
                  },
                }}
              >
                <LogoutIcon sx={{ mr: 1 }} />
                Logout
              </MenuItem>
            </Menu>
          </Box>
        </Toolbar>
      </AppBar>

      {/* Main Content */}
      <Box 
        component="main" 
        sx={{
          minHeight: 'calc(100vh - 80px)',
          p: 3,
        }}
      >
        {currentView === 'exercises' && (
          <ExerciseSelector onExerciseSelect={handleExerciseSelect} />
        )}
        
        {currentView === 'monitor' && selectedExercise && (
          <ExerciseMonitor 
            selectedExercise={selectedExercise} 
            onBack={handleBackToExercises}
          />
        )}
        
        {currentView === 'dashboard' && (
          <Dashboard />
        )}
        
        {currentView === 'debug' && (
          <MediaPipeDebug />
        )}
      </Box>
      
      {/* Floating Chatbot - Only show when user is logged in */}
      <FloatingChatbot />
    </Box>
  );
};

const App: React.FC = () => {
  return (
    <ThemeProvider theme={theme}>
      <CssBaseline />
      <AuthProvider>
        <AppContent />
      </AuthProvider>
    </ThemeProvider>
  );
};

export default App;
