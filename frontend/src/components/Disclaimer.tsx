import React, { useState } from 'react';
import {
  Box,
  Typography,
  Button,
  Paper,
  Container,
  Alert,
  Fade,
  Slide,
  IconButton,
  Divider,
  Stack,
  Chip
} from '@mui/material';
import WarningAmberIcon from '@mui/icons-material/WarningAmber';
import CheckCircleIcon from '@mui/icons-material/CheckCircle';
import SecurityIcon from '@mui/icons-material/Security';
import HealthAndSafetyIcon from '@mui/icons-material/HealthAndSafety';
import FitnessCenterIcon from '@mui/icons-material/FitnessCenter';
import ArrowForwardIcon from '@mui/icons-material/ArrowForward';

interface DisclaimerProps {
  onAccept: () => void;
}

const Disclaimer: React.FC<DisclaimerProps> = ({ onAccept }) => {
  const [animate, setAnimate] = useState(false);

  React.useEffect(() => {
    setAnimate(true);
  }, []);

  return (
    <Box
      sx={{
        minHeight: '100vh',
        background: `
          radial-gradient(circle at 20% 80%, rgba(120, 119, 198, 0.3) 0%, transparent 50%),
          radial-gradient(circle at 80% 20%, rgba(255, 119, 198, 0.15) 0%, transparent 50%),
          linear-gradient(135deg, #667eea 0%, #764ba2 50%, #f093fb 100%)
        `,
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        py: 4,
        px: 2,
        position: 'relative',
        overflow: 'hidden',
      }}
    >
      {/* Animated Background Elements */}
      <Box
        sx={{
          position: 'absolute',
          top: '10%',
          right: '15%',
          width: 100,
          height: 100,
          borderRadius: '50%',
          background: 'rgba(255, 255, 255, 0.1)',
          animation: 'float 6s ease-in-out infinite',
          '@keyframes float': {
            '0%, 100%': { transform: 'translateY(0px)' },
            '50%': { transform: 'translateY(-20px)' },
          },
        }}
      />
      <Box
        sx={{
          position: 'absolute',
          bottom: '20%',
          left: '10%',
          width: 60,
          height: 60,
          borderRadius: '50%',
          background: 'rgba(255, 255, 255, 0.08)',
          animation: 'float 8s ease-in-out infinite reverse',
        }}
      />

      <Container maxWidth="md">
        <Fade in={animate} timeout={1000}>
          <Paper
            elevation={0}
            sx={{
              background: 'rgba(255, 255, 255, 0.95)',
              backdropFilter: 'blur(30px)',
              border: '1px solid rgba(255, 255, 255, 0.3)',
              borderRadius: 6,
              overflow: 'hidden',
              boxShadow: '0 32px 64px rgba(0, 0, 0, 0.2)',
            }}
          >
            {/* Header Section */}
            <Box
              sx={{
                background: 'linear-gradient(135deg, #ff6b6b 0%, #ffd93d 100%)',
                p: 4,
                textAlign: 'center',
                position: 'relative',
              }}
            >
              <Box
                sx={{
                  width: 100,
                  height: 100,
                  borderRadius: '50%',
                  background: 'rgba(255, 255, 255, 0.2)',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  margin: '0 auto',
                  mb: 3,
                  border: '3px solid rgba(255, 255, 255, 0.3)',
                  animation: 'pulse 2s ease-in-out infinite',
                  '@keyframes pulse': {
                    '0%, 100%': { transform: 'scale(1)' },
                    '50%': { transform: 'scale(1.05)' },
                  },
                }}
              >
                <WarningAmberIcon 
                  sx={{ 
                    fontSize: 50, 
                    color: 'white',
                    filter: 'drop-shadow(0 4px 8px rgba(0,0,0,0.2))',
                  }} 
                />
              </Box>
              
              <Typography 
                variant="h3" 
                component="h1" 
                sx={{ 
                  fontWeight: 800,
                  color: 'white',
                  mb: 1,
                  textShadow: '0 4px 8px rgba(0,0,0,0.3)',
                  letterSpacing: '-0.02em',
                }}
              >
                Important Safety Notice
              </Typography>
              
              <Typography 
                variant="h6" 
                sx={{ 
                  color: 'rgba(255, 255, 255, 0.9)',
                  fontWeight: 500,
                  textShadow: '0 2px 4px rgba(0,0,0,0.2)',
                }}
              >
                Please read carefully before proceeding
              </Typography>
            </Box>

            {/* Content Section */}
            <Box sx={{ p: 5 }}>
              {/* Safety Icons */}
              <Stack direction="row" spacing={2} justifyContent="center" sx={{ mb: 4 }}>
                <Chip
                  icon={<SecurityIcon />}
                  label="Safety First"
                  variant="outlined"
                  sx={{
                    borderColor: '#667eea',
                    color: '#667eea',
                    fontWeight: 600,
                  }}
                />
                <Chip
                  icon={<HealthAndSafetyIcon />}
                  label="Health Priority"
                  variant="outlined"
                  sx={{
                    borderColor: '#38a169',
                    color: '#38a169',
                    fontWeight: 600,
                  }}
                />
                <Chip
                  icon={<FitnessCenterIcon />}
                  label="Exercise Monitoring"
                  variant="outlined"
                  sx={{
                    borderColor: '#764ba2',
                    color: '#764ba2',
                    fontWeight: 600,
                  }}
                />
              </Stack>

              {/* Main Disclaimer Text */}
              <Paper
                elevation={0}
                sx={{
                  background: 'linear-gradient(135deg, #fff5f5 0%, #fef7f0 100%)',
                  border: '2px solid #fed7d7',
                  borderRadius: 3,
                  p: 4,
                  mb: 4,
                  position: 'relative',
                }}
              >
                <Box
                  sx={{
                    position: 'absolute',
                    top: -15,
                    left: 30,
                    background: '#fff',
                    px: 2,
                    py: 0.5,
                    borderRadius: 2,
                    border: '2px solid #fed7d7',
                  }}
                >
                  <Typography variant="caption" sx={{ fontWeight: 700, color: '#e53e3e' }}>
                    DISCLAIMER
                  </Typography>
                </Box>

                <Stack spacing={3} sx={{ mt: 1 }}>
                  <Box sx={{ display: 'flex', alignItems: 'flex-start', gap: 2 }}>
                    <Box
                      sx={{
                        width: 8,
                        height: 8,
                        borderRadius: '50%',
                        background: '#e53e3e',
                        mt: 1,
                        flexShrink: 0,
                      }}
                    />
                    <Typography variant="body1" sx={{ lineHeight: 1.7, fontWeight: 500 }}>
                      This application is for <strong>informational and educational purposes only</strong> and does not replace professional medical advice, diagnosis, or treatment.
                    </Typography>
                  </Box>

                  <Box sx={{ display: 'flex', alignItems: 'flex-start', gap: 2 }}>
                    <Box
                      sx={{
                        width: 8,
                        height: 8,
                        borderRadius: '50%',
                        background: '#e53e3e',
                        mt: 1,
                        flexShrink: 0,
                      }}
                    />
                    <Typography variant="body1" sx={{ lineHeight: 1.7, fontWeight: 500 }}>
                      Use this exercise monitoring system <strong>at your own risk</strong> and only with prior approval from a certified healthcare professional or fitness trainer.
                    </Typography>
                  </Box>

                  <Box sx={{ display: 'flex', alignItems: 'flex-start', gap: 2 }}>
                    <Box
                      sx={{
                        width: 8,
                        height: 8,
                        borderRadius: '50%',
                        background: '#e53e3e',
                        mt: 1,
                        flexShrink: 0,
                      }}
                    />
                    <Typography variant="body1" sx={{ lineHeight: 1.7, fontWeight: 500 }}>
                      If you experience any pain, discomfort, or unusual symptoms during exercise, <strong>stop immediately</strong> and consult a healthcare provider.
                    </Typography>
                  </Box>
                </Stack>
              </Paper>

              <Divider sx={{ my: 3 }} />

              {/* Agreement Section */}
              <Box sx={{ textAlign: 'center' }}>
                <Typography variant="h6" sx={{ mb: 3, fontWeight: 600, color: '#2d3748' }}>
                  By proceeding, you acknowledge that you have read, understood, and accept these terms.
                </Typography>

                <Slide direction="up" in={animate} timeout={1500}>
                  <Button
                    variant="contained"
                    size="large"
                    onClick={onAccept}
                    endIcon={<ArrowForwardIcon />}
                    sx={{
                      py: 2,
                      px: 6,
                      fontSize: '1.2rem',
                      fontWeight: 700,
                      borderRadius: 4,
                      textTransform: 'none',
                      background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
                      boxShadow: '0 8px 32px rgba(102, 126, 234, 0.4)',
                      position: 'relative',
                      overflow: 'hidden',
                      '&:hover': {
                        background: 'linear-gradient(135deg, #5a6fd8 0%, #6a4190 100%)',
                        transform: 'translateY(-2px)',
                        boxShadow: '0 12px 40px rgba(102, 126, 234, 0.6)',
                      },
                      '&:before': {
                        content: '""',
                        position: 'absolute',
                        top: 0,
                        left: '-100%',
                        width: '100%',
                        height: '100%',
                        background: 'linear-gradient(90deg, transparent, rgba(255,255,255,0.2), transparent)',
                        transition: 'left 0.7s',
                      },
                      '&:hover:before': {
                        left: '100%',
                      },
                      transition: 'all 0.3s cubic-bezier(0.4, 0, 0.2, 1)',
                    }}
                  >
                    I Understand & Accept
                  </Button>
                </Slide>
              </Box>

              {/* Footer Note */}
              <Box sx={{ mt: 4, textAlign: 'center' }}>
                <Typography 
                  variant="caption" 
                  sx={{ 
                    color: 'text.secondary',
                    fontStyle: 'italic',
                    fontSize: '0.9rem',
                    opacity: 0.8,
                  }}
                >
                  This disclaimer is shown every time you open or log into the application for your safety
                </Typography>
              </Box>
            </Box>
          </Paper>
        </Fade>
      </Container>
    </Box>
  );
};

export default Disclaimer;
