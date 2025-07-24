import React, { useState, useEffect, useRef } from 'react';
import {
  Box,
  Paper,
  TextField,
  Button,
  Typography,
  Avatar,
  IconButton,
  Chip,
  CircularProgress,
  Fade,
  Divider
} from '@mui/material';
import SendIcon from '@mui/icons-material/Send';
import SmartToyIcon from '@mui/icons-material/SmartToy';
import PersonIcon from '@mui/icons-material/Person';
import CloseIcon from '@mui/icons-material/Close';
import MinimizeIcon from '@mui/icons-material/Minimize';
import { useAuth } from '../contexts/AuthContext';

interface ChatMessage {
  role: 'user' | 'assistant';
  content: string;
  timestamp: Date;
}

interface ChatbotProps {
  isOpen: boolean;
  onClose: () => void;
  onMinimize: () => void;
}

const Chatbot: React.FC<ChatbotProps> = ({ isOpen, onClose, onMinimize }) => {
  const { currentUser } = useAuth();
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [inputMessage, setInputMessage] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [isTyping, setIsTyping] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  
  const CHATBOT_API_URL = "http://localhost:5000/chatbot";
  const STORAGE_KEY = `chatbot_messages_${currentUser?.uid || 'anonymous'}`;

  // Function to format message content for better display
  const formatMessage = (content: string): string => {
    return content
      // Convert **text** to bold
      .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
      // Convert 🏋️ **TITLE** to headers
      .replace(/^(🏋️|💪|🥗|🎯|📊|🍽️|💧|🔥|🏃|🧘|🆘|💡|📋|📝|🚀|⚡|🩺|🔧|💯|🎉)[\s]*\*\*(.*?)\*\*/gm, '<h4>$1 $2</h4>')
      // Convert **SECTION:** to subheaders
      .replace(/^\*\*(.*?):\*\*/gm, '<h5>$1:</h5>')
      // Convert bullet points
      .replace(/^[-•]\s/gm, '• ')
      .replace(/^✅\s/gm, '✅ ')
      .replace(/^🎯\s/gm, '🎯 ')
      .replace(/^🔥\s/gm, '🔥 ')
      .replace(/^💪\s/gm, '💪 ')
      .replace(/^🏋️\s/gm, '🏋️ ')
      .replace(/^🥗\s/gm, '🥗 ')
      // Convert newlines to HTML breaks
      .replace(/\n\n/g, '<br><br>')
      .replace(/\n/g, '<br>')
      // Style exercise lists
      .replace(/^(Day \d+.*?):/gm, '<strong style="color: #1976d2;">$1:</strong>')
      .replace(/^(\d+\.\s.*?):/gm, '<strong>$1:</strong>')
      // Style meal plans
      .replace(/^(Breakfast|Lunch|Dinner|Snack|Pre-workout|Post-workout|Before bed):/gm, '<strong style="color: #dc004e;">$1:</strong>')
      // Style exercise names with sets/reps
      .replace(/^-\s(.*?):\s(\d+.*)/gm, '• <strong>$1:</strong> $2');
  };

  // Auto-scroll to bottom when new messages arrive
  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  // Load chat history from localStorage on component mount
  useEffect(() => {
    if (currentUser?.uid) {
      loadChatHistory();
    }
  }, [currentUser?.uid]);

  const loadChatHistory = () => {
    try {
      const storedMessages = localStorage.getItem(STORAGE_KEY);
      if (storedMessages) {
        const parsedMessages = JSON.parse(storedMessages).map((msg: any) => ({
          ...msg,
          timestamp: new Date(msg.timestamp)
        }));
        setMessages(parsedMessages);
      } else {
        // Add welcome message for new users
        const welcomeMessage: ChatMessage = {
          role: 'assistant',
          content: "Hi! I'm your fitness AI assistant. I can help you with exercise tips, answer questions about your workouts, and provide guidance on your fitness journey. How can I help you today?",
          timestamp: new Date()
        };
        setMessages([welcomeMessage]);
        saveChatHistory([welcomeMessage]);
      }
    } catch (error) {
      console.error('Error loading chat history:', error);
    }
  };

  const saveChatHistory = (messagesToSave: ChatMessage[]) => {
    try {
      localStorage.setItem(STORAGE_KEY, JSON.stringify(messagesToSave));
    } catch (error) {
      console.error('Error saving chat history:', error);
    }
  };

  const sendMessage = async () => {
    if (!inputMessage.trim() || isLoading || !currentUser?.uid) return;

    const userMessage: ChatMessage = {
      role: 'user',
      content: inputMessage.trim(),
      timestamp: new Date()
    };

    // Add user message to state
    const updatedMessages = [...messages, userMessage];
    setMessages(updatedMessages);
    setInputMessage('');
    setIsLoading(true);
    setIsTyping(true);

    try {
      // Prepare messages for API (convert to OpenAI format)
      const apiMessages = updatedMessages.map(msg => ({
        role: msg.role,
        content: msg.content
      }));

      // Send to backend
      const response = await fetch(CHATBOT_API_URL, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          uid: currentUser.uid,
          messages: apiMessages
        }),
      });

      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
      }

      const data = await response.json();
      
      // Add assistant response
      const assistantMessage: ChatMessage = {
        role: 'assistant',
        content: data.reply || 'Sorry, I encountered an error. Please try again.',
        timestamp: new Date()
      };

      const finalMessages = [...updatedMessages, assistantMessage];
      setMessages(finalMessages);
      saveChatHistory(finalMessages);

    } catch (error) {
      console.error('Chatbot API error:', error);
      
      // Add error message
      const errorMessage: ChatMessage = {
        role: 'assistant',
        content: 'Sorry, I\'m having trouble connecting right now. Please check your internet connection and try again.',
        timestamp: new Date()
      };
      
      const finalMessages = [...updatedMessages, errorMessage];
      setMessages(finalMessages);
      saveChatHistory(finalMessages);
    } finally {
      setIsLoading(false);
      setIsTyping(false);
    }
  };

  const handleKeyPress = (event: React.KeyboardEvent) => {
    if (event.key === 'Enter' && !event.shiftKey) {
      event.preventDefault();
      sendMessage();
    }
  };

  const clearChat = () => {
    const welcomeMessage: ChatMessage = {
      role: 'assistant',
      content: "Chat cleared! How can I help you today?",
      timestamp: new Date()
    };
    setMessages([welcomeMessage]);
    saveChatHistory([welcomeMessage]);
  };

  const formatTime = (date: Date) => {
    return date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
  };

  if (!isOpen) return null;

  return (
    <Fade in={isOpen}>
      <Paper
        elevation={8}
        sx={{
          position: 'fixed',
          bottom: 20,
          right: 20,
          width: 400,
          height: 600,
          display: 'flex',
          flexDirection: 'column',
          zIndex: 1300,
          borderRadius: 2,
          overflow: 'hidden'
        }}
      >
        {/* Header */}
        <Box
          sx={{
            bgcolor: 'primary.main',
            color: 'white',
            p: 2,
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'space-between'
          }}
        >
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
            <Avatar sx={{ bgcolor: 'primary.dark', width: 32, height: 32 }}>
              <SmartToyIcon fontSize="small" />
            </Avatar>
            <Box>
              <Typography variant="h6" sx={{ fontSize: '1rem', fontWeight: 600 }}>
                Fitness AI Assistant
              </Typography>
              <Typography variant="caption" sx={{ opacity: 0.8 }}>
                {isTyping ? 'Typing...' : 'Online'}
              </Typography>
            </Box>
          </Box>
          <Box>
            <IconButton size="small" onClick={onMinimize} sx={{ color: 'white', mr: 0.5 }}>
              <MinimizeIcon fontSize="small" />
            </IconButton>
            <IconButton size="small" onClick={onClose} sx={{ color: 'white' }}>
              <CloseIcon fontSize="small" />
            </IconButton>
          </Box>
        </Box>

        {/* Messages Area */}
        <Box
          sx={{
            flex: 1,
            overflow: 'auto',
            p: 1,
            backgroundColor: '#f5f5f5',
            display: 'flex',
            flexDirection: 'column',
            gap: 1
          }}
        >
          {messages.map((message, index) => (
            <Box
              key={index}
              sx={{
                display: 'flex',
                justifyContent: message.role === 'user' ? 'flex-end' : 'flex-start',
                mb: 1
              }}
            >
              <Box
                sx={{
                  display: 'flex',
                  alignItems: 'flex-start',
                  gap: 1,
                  maxWidth: '80%',
                  flexDirection: message.role === 'user' ? 'row-reverse' : 'row'
                }}
              >
                <Avatar
                  sx={{
                    bgcolor: message.role === 'user' ? 'primary.main' : 'secondary.main',
                    width: 28,
                    height: 28,
                    mt: 0.5
                  }}
                >
                  {message.role === 'user' ? (
                    <PersonIcon fontSize="small" />
                  ) : (
                    <SmartToyIcon fontSize="small" />
                  )}
                </Avatar>
                <Box>
                  <Paper
                    elevation={1}
                    sx={{
                      p: 1.5,
                      bgcolor: message.role === 'user' ? 'primary.main' : 'white',
                      color: message.role === 'user' ? 'white' : 'text.primary',
                      borderRadius: 2,
                      borderTopLeftRadius: message.role === 'user' ? 2 : 0.5,
                      borderTopRightRadius: message.role === 'user' ? 0.5 : 2
                    }}
                  >
                    <Box
                      sx={{
                        wordBreak: 'break-word',
                        lineHeight: 1.4,
                        '& h4': {
                          fontSize: '1rem',
                          fontWeight: 'bold',
                          margin: '8px 0 4px 0',
                          color: message.role === 'user' ? 'white' : 'primary.main'
                        },
                        '& h5': {
                          fontSize: '0.9rem',
                          fontWeight: 'bold',
                          margin: '6px 0 4px 0',
                          color: message.role === 'user' ? 'white' : 'secondary.main'
                        },
                        '& ul': {
                          margin: '4px 0',
                          paddingLeft: '16px'
                        },
                        '& li': {
                          margin: '2px 0'
                        },
                        '& strong': {
                          fontWeight: 600
                        }
                      }}
                      dangerouslySetInnerHTML={{
                        __html: formatMessage(message.content)
                      }}
                    />
                  </Paper>
                  <Typography
                    variant="caption"
                    sx={{
                      color: 'text.secondary',
                      fontSize: '0.7rem',
                      mt: 0.5,
                      display: 'block',
                      textAlign: message.role === 'user' ? 'right' : 'left'
                    }}
                  >
                    {formatTime(message.timestamp)}
                  </Typography>
                </Box>
              </Box>
            </Box>
          ))}
          
          {/* Loading indicator */}
          {isLoading && (
            <Box sx={{ display: 'flex', justifyContent: 'flex-start', mb: 1 }}>
              <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                <Avatar sx={{ bgcolor: 'secondary.main', width: 28, height: 28 }}>
                  <SmartToyIcon fontSize="small" />
                </Avatar>
                <Paper
                  elevation={1}
                  sx={{
                    p: 2,
                    bgcolor: 'white',
                    borderRadius: 2,
                    borderTopLeftRadius: 0.5
                  }}
                >
                  <CircularProgress size={16} />
                </Paper>
              </Box>
            </Box>
          )}
          
          <div ref={messagesEndRef} />
        </Box>

        <Divider />

        {/* Quick Actions */}
        <Box sx={{ p: 1, display: 'flex', gap: 0.5, flexWrap: 'wrap' }}>
          <Chip
            label="Best chest exercises?"
            size="small"
            variant="outlined"
            onClick={() => setInputMessage("What are the best chest exercises I can do at home?")}
            sx={{ fontSize: '0.7rem' }}
          />
          <Chip
            label="How to gain weight?"
            size="small"
            variant="outlined"
            onClick={() => setInputMessage("I'm skinny and want to gain weight. What should I do?")}
            sx={{ fontSize: '0.7rem' }}
          />
          <Chip
            label="Make me a plan"
            size="small"
            variant="outlined"
            onClick={() => setInputMessage("Can you make me a plan? I'm 22 years old, 5'6\", 55kg and want to gain 10kg in 3 months")}
            sx={{ fontSize: '0.7rem' }}
          />
          <Chip
            label="Clear chat"
            size="small"
            variant="outlined"
            onClick={clearChat}
            sx={{ fontSize: '0.7rem' }}
          />
        </Box>

        {/* Input Area */}
        <Box
          sx={{
            p: 2,
            backgroundColor: 'white',
            display: 'flex',
            gap: 1,
            alignItems: 'flex-end'
          }}
        >
          <TextField
            fullWidth
            multiline
            maxRows={3}
            placeholder="Ask me about fitness, exercises, or your workout..."
            value={inputMessage}
            onChange={(e) => setInputMessage(e.target.value)}
            onKeyPress={handleKeyPress}
            disabled={isLoading}
            size="small"
            sx={{
              '& .MuiOutlinedInput-root': {
                borderRadius: 2
              }
            }}
          />
          <Button
            variant="contained"
            onClick={sendMessage}
            disabled={!inputMessage.trim() || isLoading}
            sx={{
              minWidth: 'auto',
              p: 1,
              borderRadius: 2
            }}
          >
            <SendIcon fontSize="small" />
          </Button>
        </Box>
      </Paper>
    </Fade>
  );
};

export default Chatbot;
