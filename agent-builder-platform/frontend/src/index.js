import React from 'react';
import ReactDOM from 'react-dom/client';
import { ThemeProvider, createTheme } from '@mui/material/styles';
import CssBaseline from '@mui/material/CssBaseline';
import App from './App.jsx';

// Create dark theme with green accents
const theme = createTheme({
  palette: {
    mode: 'dark',
    primary: {
      main: '#00ff88', // Bright green
    },
    secondary: {
      main: '#ff6b6b', // Coral for accents
    },
    background: {
      default: '#0a0a0a',
      paper: '#1a1a1a',
    },
    text: {
      primary: '#ffffff',
      secondary: '#b0b0b0',
    },
  },
  typography: {
    fontFamily: '"Inter", "Roboto", "Helvetica", "Arial", sans-serif',
    h1: {
      fontSize: '3.5rem',
      fontWeight: 700,
      background: 'linear-gradient(135deg, #00ff88 0%, #ff6b6b 100%)',
      WebkitBackgroundClip: 'text',
      WebkitTextFillColor: 'transparent',
      backgroundClip: 'text',
    },
    h2: {
      fontSize: '1.5rem',
      fontWeight: 500,
      color: '#ffffff',
    },
    body1: {
      fontSize: '1.1rem',
      color: '#b0b0b0',
      lineHeight: 1.6,
    },
  },
  components: {
    MuiButton: {
      styleOverrides: {
        root: {
          borderRadius: '12px',
          textTransform: 'none',
          fontSize: '1.1rem',
          fontWeight: 600,
          padding: '12px 32px',
          background: 'linear-gradient(135deg, #00ff88 0%, #ff6b6b 100%)',
          color: '#000000',
          '&:hover': {
            background: 'linear-gradient(135deg, #00cc6a 0%, #ff5252 100%)',
            transform: 'translateY(-2px)',
            boxShadow: '0 8px 25px rgba(0, 255, 136, 0.3)',
          },
          transition: 'all 0.3s ease',
        },
      },
    },
    MuiCard: {
      styleOverrides: {
        root: {
          background: 'rgba(26, 26, 26, 0.8)',
          backdropFilter: 'blur(10px)',
          border: '1px solid rgba(0, 255, 136, 0.2)',
          borderRadius: '16px',
          padding: '24px',
          transition: 'all 0.3s ease',
          '&:hover': {
            border: '1px solid rgba(0, 255, 136, 0.4)',
            transform: 'translateY(-4px)',
            boxShadow: '0 12px 40px rgba(0, 255, 136, 0.1)',
          },
        },
      },
    },
  },
});

const root = ReactDOM.createRoot(document.getElementById('root'));
root.render(
  <React.StrictMode>
    <ThemeProvider theme={theme}>
      <CssBaseline />
      <App />
    </ThemeProvider>
  </React.StrictMode>
);
