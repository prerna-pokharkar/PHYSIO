import React, { useState } from 'react';
import { Fab, Badge, Zoom } from '@mui/material';
import ChatIcon from '@mui/icons-material/Chat';
import SmartToyIcon from '@mui/icons-material/SmartToy';
import Chatbot from './Chatbot';

interface FloatingChatbotProps {
  hasNewMessages?: boolean;
}

const FloatingChatbot: React.FC<FloatingChatbotProps> = ({ hasNewMessages = false }) => {
  const [isOpen, setIsOpen] = useState(false);
  const [isMinimized, setIsMinimized] = useState(false);

  const handleToggle = () => {
    if (isMinimized) {
      setIsMinimized(false);
    } else {
      setIsOpen(!isOpen);
    }
  };

  const handleClose = () => {
    setIsOpen(false);
    setIsMinimized(false);
  };

  const handleMinimize = () => {
    setIsOpen(false);
    setIsMinimized(true);
  };

  return (
    <>
      {/* Floating Action Button */}
      <Zoom in={!isOpen}>
        <Fab
          color="primary"
          onClick={handleToggle}
          sx={{
            position: 'fixed',
            bottom: 24,
            right: 24,
            zIndex: 1200,
            width: 64,
            height: 64,
            background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
            boxShadow: '0px 8px 24px rgba(102, 126, 234, 0.4)',
            '&:hover': {
              transform: 'scale(1.1)',
              boxShadow: '0px 12px 32px rgba(102, 126, 234, 0.6)',
              background: 'linear-gradient(135deg, #5a6fd8 0%, #6a4190 100%)',
            },
            transition: 'all 0.3s cubic-bezier(0.4, 0, 0.2, 1)',
            '&:before': {
              content: '""',
              position: 'absolute',
              top: -2,
              left: -2,
              right: -2,
              bottom: -2,
              background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
              borderRadius: '50%',
              zIndex: -1,
              opacity: 0.5,
              filter: 'blur(8px)',
            },
          }}
        >
          <Badge
            badgeContent={hasNewMessages ? '!' : 0}
            color="error"
            sx={{
              '& .MuiBadge-badge': {
                fontSize: '0.75rem',
                minWidth: 18,
                height: 18,
                background: 'linear-gradient(135deg, #ff6b6b 0%, #ee5a52 100%)',
                boxShadow: '0px 2px 8px rgba(238, 90, 82, 0.4)',
              },
            }}
          >
            {isMinimized ? (
              <ChatIcon sx={{ fontSize: 28, color: 'white' }} />
            ) : (
              <SmartToyIcon sx={{ fontSize: 28, color: 'white' }} />
            )}
          </Badge>
        </Fab>
      </Zoom>

      {/* Minimized indicator */}
      {isMinimized && (
        <Zoom in={isMinimized}>
          <Fab
            size="small"
            color="secondary"
            onClick={handleToggle}
            sx={{
              position: 'fixed',
              bottom: 90,
              right: 20,
              zIndex: 1200,
              opacity: 0.8,
              '&:hover': {
                opacity: 1,
              },
            }}
          >
            <ChatIcon fontSize="small" />
          </Fab>
        </Zoom>
      )}

      {/* Chatbot Component */}
      <Chatbot
        isOpen={isOpen}
        onClose={handleClose}
        onMinimize={handleMinimize}
      />
    </>
  );
};

export default FloatingChatbot;
