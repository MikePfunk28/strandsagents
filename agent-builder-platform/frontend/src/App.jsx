import React, { useState, useEffect } from 'react';
import {
  Box,
  Container,
  Typography,
  Card,
  CardContent,
  Button,
  AppBar,
  Toolbar,
  IconButton,
  Dialog,
  DialogTitle,
  DialogContent,
  TextField,
  Select,
  MenuItem,
  FormControl,
  InputLabel,
  Grid,
  Chip,
  LinearProgress,
  Alert
} from '@mui/material';
import {
  Settings as SettingsIcon,
  ArrowBack as ArrowBackIcon,
  AccessTime as TimeIcon,
  Security as SecurityIcon,
  AttachMoney as MoneyIcon,
  AutoAwesome as SparkleIcon
} from '@mui/icons-material';
import { motion } from 'framer-motion';
import axios from 'axios';

const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';

function App() {
  const [currentView, setCurrentView] = useState('landing'); // landing, consultation, building
  const [sessionId, setSessionId] = useState(null);
  const [projectId, setProjectId] = useState(null);
  const [currentPhase, setCurrentPhase] = useState('initialization');
  const [progress, setProgress] = useState(0);
  const [userInput, setUserInput] = useState('');
  const [agentResponse, setAgentResponse] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [wsConnection, setWsConnection] = useState(null);

  // Agent creation form state
  const [agentForm, setAgentForm] = useState({
    name: '',
    type: 'chatbot',
    description: '',
    systemPrompt: '',
    tools: [],
    awsServices: []
  });

  // Initialize session on component mount
  useEffect(() => {
    initializeSession();
    setupWebSocket();

    return () => {
      if (wsConnection) {
        wsConnection.close();
      }
    };
  }, []);

  const initializeSession = async () => {
    try {
      const response = await axios.post(`${API_BASE_URL}/sessions/create`, {
        experience_level: 'intermediate'
      });
      setSessionId(response.data.session_id);
    } catch (error) {
      console.error('Failed to initialize session:', error);
      setError('Failed to initialize session. Please refresh the page.');
    }
  };

  const setupWebSocket = () => {
    if (sessionId) {
      const ws = new WebSocket(`ws://localhost:8000/ws/${sessionId}`);

      ws.onopen = () => {
        console.log('WebSocket connected');
        setWsConnection(ws);
      };

      ws.onmessage = (event) => {
        const data = JSON.parse(event.data);
        if (data.type === 'workflow_update') {
          setCurrentPhase(data.current_phase);
          setProgress(data.progress_percentage);
        }
      };

      ws.onerror = (error) => {
        console.error('WebSocket error:', error);
      };

      ws.onclose = () => {
        console.log('WebSocket connection closed');
        setWsConnection(null);
      };
    }
  };

  const startConsultation = async () => {
    if (!userInput.trim()) {
      setError('Please describe what you want to build');
      return;
    }

    setLoading(true);
    setError('');

    try {
      // Create project
      const projectResponse = await axios.post(`${API_BASE_URL}/projects/create`, {
        project_name: agentForm.name || 'My Agent',
        use_case: agentForm.type,
        description: userInput,
        experience_level: 'intermediate'
      }, {
        params: { session_id: sessionId }
      });

      setProjectId(projectResponse.data.project_id);
      setCurrentView('consultation');

      // Get initial response from AWS Solutions Architect
      const response = await axios.post(`${API_BASE_URL}/workflow/requirements`, {
        project_id: projectResponse.data.project_id,
        user_input: userInput
      }, {
        params: { session_id: sessionId }
      });

      setAgentResponse(response.data.content);
      setCurrentPhase('requirements');

    } catch (error) {
      console.error('Failed to start consultation:', error);
      setError('Failed to start consultation. Please try again.');
    } finally {
      setLoading(false);
    }
  };

  const submitResponse = async (feedback: string) => {
    if (!feedback.trim()) return;

    setLoading(true);

    try {
      let response;
      if (currentPhase === 'requirements') {
        response = await axios.post(`${API_BASE_URL}/workflow/architecture`, {
          project_id: projectId,
          user_feedback: feedback
        }, {
          params: { session_id: sessionId }
        });
        setCurrentPhase('architecture');
      } else if (currentPhase === 'architecture') {
        response = await axios.post(`${API_BASE_URL}/workflow/implementation`, {
          project_id: projectId,
          user_feedback: feedback
        }, {
          params: { session_id: sessionId }
        });
        setCurrentPhase('implementation');
      } else {
        response = await axios.post(`${API_BASE_URL}/workflow/testing`, {
          project_id: projectId,
          user_feedback: feedback
        }, {
          params: { session_id: sessionId }
        });
        setCurrentPhase('testing');
      }

      setAgentResponse(response.data.content);

    } catch (error) {
      console.error('Failed to submit response:', error);
      setError('Failed to submit response. Please try again.');
    } finally {
      setLoading(false);
    }
  };

  const generateAgent = async () => {
    try {
      setLoading(true);

      const exportResponse = await axios.post(`${API_BASE_URL}/export/generate`, {
        project_id: projectId,
        export_format: 'python',
        include_documentation: true,
        include_tests: true,
        deployment_ready: true
      });

      setCurrentView('building');
      setProgress(100);

      // In a real implementation, this would trigger the actual agent generation
      setTimeout(() => {
        setCurrentView('complete');
      }, 3000);

    } catch (error) {
      console.error('Failed to generate agent:', error);
      setError('Failed to generate agent. Please try again.');
    } finally {
      setLoading(false);
    }
  };

  const resetToLanding = () => {
    setCurrentView('landing');
    setProjectId(null);
    setCurrentPhase('initialization');
    setProgress(0);
    setUserInput('');
    setAgentResponse('');
    setError('');
  };

  // Landing page view
  if (currentView === 'landing') {
    return (
      <Box sx={{
        minHeight: '100vh',
        background: 'linear-gradient(135deg, #0a0a0a 0%, #1a1a1a 100%)',
        color: 'white'
      }}>
        {/* Header */}
        <AppBar position="static" elevation={0} sx={{ background: 'transparent' }}>
          <Toolbar>
            <IconButton
              edge="start"
              color="inherit"
              onClick={resetToLanding}
              sx={{ mr: 2 }}
            >
              <ArrowBackIcon />
            </IconButton>
            <Typography variant="h6" component="div" sx={{ flexGrow: 1 }}>
              Agent Builder Platform
            </Typography>
            <IconButton color="inherit">
              <SettingsIcon />
            </IconButton>
          </Toolbar>
        </AppBar>

        {/* Main Content */}
        <Container maxWidth="lg" sx={{ py: 8 }}>
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8 }}
          >
            {/* Hero Section */}
            <Box textAlign="center" sx={{ mb: 8 }}>
              <motion.div
                animate={{
                  rotate: [0, 5, -5, 0],
                  scale: [1, 1.1, 1]
                }}
                transition={{
                  duration: 2,
                  repeat: Infinity,
                  repeatType: "reverse"
                }}
                style={{ display: 'inline-block', marginBottom: '2rem' }}
              >
                <SparkleIcon sx={{ fontSize: 80, color: '#00ff88' }} />
              </motion.div>

              <Typography variant="h1" component="h1" gutterBottom>
                Agent Builder Platform
              </Typography>

              <Typography variant="h2" component="h2" sx={{ mb: 4, color: '#b0b0b0' }}>
                Build production-ready AI agents in 30-45 minutes
              </Typography>

              <Typography variant="body1" sx={{ mb: 6, maxWidth: 600, mx: 'auto' }}>
                Expert AI consultants guide you through requirements, architecture, implementation,
                and deployment with confidence and precision.
              </Typography>

              {/* Feature Cards */}
              <Grid container spacing={4} sx={{ mb: 8 }}>
                <Grid item xs={12} md={4}>
                  <motion.div
                    whileHover={{ scale: 1.05 }}
                    whileTap={{ scale: 0.95 }}
                  >
                    <Card sx={{ height: '100%', textAlign: 'center' }}>
                      <CardContent>
                        <TimeIcon sx={{ fontSize: 48, color: '#00ff88', mb: 2 }} />
                        <Typography variant="h3" gutterBottom>
                          30-45 Minutes
                        </Typography>
                        <Typography variant="body2">
                          From idea to production-ready agent
                        </Typography>
                      </CardContent>
                    </Card>
                  </motion.div>
                </Grid>

                <Grid item xs={12} md={4}>
                  <motion.div
                    whileHover={{ scale: 1.05 }}
                    whileTap={{ scale: 0.95 }}
                  >
                    <Card sx={{ height: '100%', textAlign: 'center' }}>
                      <CardContent>
                        <SecurityIcon sx={{ fontSize: 48, color: '#00ff88', mb: 2 }} />
                        <Typography variant="h3" gutterBottom>
                          Built with Confidence
                        </Typography>
                        <Typography variant="body2">
                          Expert AI consultants with validated recommendations
                        </Typography>
                      </CardContent>
                    </Card>
                  </motion.div>
                </Grid>

                <Grid item xs={12} md={4}>
                  <motion.div
                    whileHover={{ scale: 1.05 }}
                    whileTap={{ scale: 0.95 }}
                  >
                    <Card sx={{ height: '100%', textAlign: 'center' }}>
                      <CardContent>
                        <MoneyIcon sx={{ fontSize: 48, color: '#00ff88', mb: 2 }} />
                        <Typography variant="h3" gutterBottom>
                          Cost Optimized
                        </Typography>
                        <Typography variant="body2">
                          Built for hackathon budgets ($16-30 total)
                        </Typography>
                      </CardContent>
                    </Card>
                  </motion.div>
                </Grid>
              </Grid>

              {/* Start Building Button */}
              <motion.div
                whileHover={{ scale: 1.05 }}
                whileTap={{ scale: 0.95 }}
              >
                <Button
                  variant="contained"
                  size="large"
                  onClick={() => setCurrentView('agent_form')}
                  sx={{
                    py: 2,
                    px: 6,
                    fontSize: '1.2rem',
                    fontWeight: 'bold',
                    borderRadius: '30px'
                  }}
                >
                  ✨ Start Building
                </Button>
              </motion.div>
            </Box>
          </motion.div>
        </Container>

        {/* Agent Creation Dialog */}
        <Dialog
          open={currentView === 'agent_form'}
          onClose={() => setCurrentView('landing')}
          maxWidth="md"
          fullWidth
          PaperProps={{
            sx: {
              background: 'rgba(26, 26, 26, 0.95)',
              backdropFilter: 'blur(10px)',
              border: '1px solid rgba(0, 255, 136, 0.2)',
              borderRadius: '16px'
            }
          }}
        >
          <DialogTitle sx={{ color: '#00ff88', textAlign: 'center' }}>
            Create Your Agent
          </DialogTitle>
          <DialogContent>
            <Grid container spacing={3} sx={{ mt: 1 }}>
              <Grid item xs={12}>
                <TextField
                  fullWidth
                  label="Describe what you want to build"
                  multiline
                  rows={4}
                  value={userInput}
                  onChange={(e) => setUserInput(e.target.value)}
                  placeholder="I want to build a customer support chatbot that can handle common inquiries..."
                  sx={{
                    '& .MuiInputBase-input': { color: 'white' },
                    '& .MuiInputLabel-root': { color: '#b0b0b0' },
                    '& .MuiOutlinedInput-root': {
                      '& fieldset': { borderColor: 'rgba(0, 255, 136, 0.3)' },
                      '&:hover fieldset': { borderColor: 'rgba(0, 255, 136, 0.5)' },
                      '&.Mui-focused fieldset': { borderColor: '#00ff88' }
                    }
                  }}
                />
              </Grid>

              <Grid item xs={12} sm={6}>
                <FormControl fullWidth>
                  <InputLabel sx={{ color: '#b0b0b0' }}>Agent Type</InputLabel>
                  <Select
                    value={agentForm.type}
                    onChange={(e) => setAgentForm({...agentForm, type: e.target.value})}
                    sx={{
                      color: 'white',
                      '& .MuiOutlinedInput-notchedOutline': { borderColor: 'rgba(0, 255, 136, 0.3)' },
                      '&:hover .MuiOutlinedInput-notchedOutline': { borderColor: 'rgba(0, 255, 136, 0.5)' },
                      '&.Mui-focused .MuiOutlinedInput-notchedOutline': { borderColor: '#00ff88' }
                    }}
                  >
                    <MenuItem value="chatbot">Chatbot</MenuItem>
                    <MenuItem value="api">API Agent</MenuItem>
                    <MenuItem value="data_processing">Data Processing</MenuItem>
                    <MenuItem value="monitoring">Monitoring</MenuItem>
                    <MenuItem value="automation">Automation</MenuItem>
                    <MenuItem value="custom">Custom</MenuItem>
                  </Select>
                </FormControl>
              </Grid>

              <Grid item xs={12} sm={6}>
                <TextField
                  fullWidth
                  label="Agent Name (Optional)"
                  value={agentForm.name}
                  onChange={(e) => setAgentForm({...agentForm, name: e.target.value})}
                  sx={{
                    '& .MuiInputBase-input': { color: 'white' },
                    '& .MuiInputLabel-root': { color: '#b0b0b0' },
                    '& .MuiOutlinedInput-root': {
                      '& fieldset': { borderColor: 'rgba(0, 255, 136, 0.3)' },
                      '&:hover fieldset': { borderColor: 'rgba(0, 255, 136, 0.5)' },
                      '&.Mui-focused fieldset': { borderColor: '#00ff88' }
                    }
                  }}
                />
              </Grid>

              <Grid item xs={12}>
                <Box sx={{ display: 'flex', gap: 2, justifyContent: 'center' }}>
                  <Button
                    variant="outlined"
                    onClick={() => setCurrentView('landing')}
                    sx={{
                      borderColor: '#00ff88',
                      color: '#00ff88',
                      '&:hover': { borderColor: '#00cc6a', backgroundColor: 'rgba(0, 255, 136, 0.1)' }
                    }}
                  >
                    Cancel
                  </Button>
                  <Button
                    variant="contained"
                    onClick={startConsultation}
                    disabled={loading || !userInput.trim()}
                    sx={{
                      background: 'linear-gradient(135deg, #00ff88 0%, #ff6b6b 100%)',
                      color: '#000000',
                      fontWeight: 'bold'
                    }}
                  >
                    {loading ? 'Starting...' : 'Begin Consultation'}
                  </Button>
                </Box>
              </Grid>
            </Grid>
          </DialogContent>
        </Dialog>
      </Box>
    );
  }

  // Consultation view
  if (currentView === 'consultation') {
    return (
      <Box sx={{
        minHeight: '100vh',
        background: 'linear-gradient(135deg, #0a0a0a 0%, #1a1a1a 100%)',
        color: 'white'
      }}>
        {/* Header */}
        <AppBar position="static" elevation={0} sx={{ background: 'transparent' }}>
          <Toolbar>
            <IconButton
              edge="start"
              color="inherit"
              onClick={resetToLanding}
              sx={{ mr: 2 }}
            >
              <ArrowBackIcon />
            </IconButton>
            <Typography variant="h6" component="div" sx={{ flexGrow: 1 }}>
              Agent Builder Platform - Consultation
            </Typography>
            <Chip
              label={`${currentPhase} - ${progress}%`}
              sx={{
                backgroundColor: '#00ff88',
                color: '#000000',
                fontWeight: 'bold'
              }}
            />
          </Toolbar>
          <LinearProgress
            variant="determinate"
            value={progress}
            sx={{
              height: '4px',
              '& .MuiLinearProgress-bar': {
                backgroundColor: '#00ff88'
              }
            }}
          />
        </AppBar>

        {/* Consultation Content */}
        <Container maxWidth="lg" sx={{ py: 4 }}>
          <Grid container spacing={4}>
            {/* Agent Response */}
            <Grid item xs={12} md={8}>
              <Card sx={{ mb: 4 }}>
                <CardContent>
                  <Typography variant="h6" sx={{ color: '#00ff88', mb: 2 }}>
                    🤖 AI Consultant Response
                  </Typography>
                  <Typography variant="body1" sx={{ lineHeight: 1.6, mb: 3 }}>
                    {agentResponse}
                  </Typography>

                  {loading && (
                    <LinearProgress sx={{ '& .MuiLinearProgress-bar': { backgroundColor: '#00ff88' } }} />
                  )}
                </CardContent>
              </Card>

              {/* User Input */}
              <Card>
                <CardContent>
                  <Typography variant="h6" sx={{ color: '#00ff88', mb: 2 }}>
                    💬 Your Response
                  </Typography>
                  <TextField
                    fullWidth
                    multiline
                    rows={4}
                    placeholder="Type your response or feedback here..."
                    value={userInput}
                    onChange={(e) => setUserInput(e.target.value)}
                    sx={{
                      mb: 2,
                      '& .MuiInputBase-input': { color: 'white' },
                      '& .MuiOutlinedInput-root': {
                        '& fieldset': { borderColor: 'rgba(0, 255, 136, 0.3)' },
                        '&:hover fieldset': { borderColor: 'rgba(0, 255, 136, 0.5)' },
                        '&.Mui-focused fieldset': { borderColor: '#00ff88' }
                      }
                    }}
                  />
                  <Box sx={{ display: 'flex', gap: 2, justifyContent: 'flex-end' }}>
                    <Button
                      variant="outlined"
                      onClick={() => setUserInput('')}
                      sx={{
                        borderColor: '#b0b0b0',
                        color: '#b0b0b0'
                      }}
                    >
                      Clear
                    </Button>
                    <Button
                      variant="contained"
                      onClick={() => submitResponse(userInput)}
                      disabled={loading || !userInput.trim()}
                      sx={{
                        background: 'linear-gradient(135deg, #00ff88 0%, #ff6b6b 100%)',
                        color: '#000000',
                        fontWeight: 'bold'
                      }}
                    >
                      {loading ? 'Processing...' : 'Submit Response'}
                    </Button>
                  </Box>
                </CardContent>
              </Card>
            </Grid>

            {/* Sidebar */}
            <Grid item xs={12} md={4}>
              <Card sx={{ mb: 4 }}>
                <CardContent>
                  <Typography variant="h6" sx={{ color: '#00ff88', mb: 2 }}>
                    📊 Consultation Progress
                  </Typography>
                  <Box sx={{ mb: 3 }}>
                    <Typography variant="body2" sx={{ mb: 1 }}>
                      Current Phase: {currentPhase}
                    </Typography>
                    <Typography variant="body2" sx={{ mb: 2 }}>
                      Progress: {progress}%
                    </Typography>
                    <LinearProgress
                      variant="determinate"
                      value={progress}
                      sx={{
                        height: '8px',
                        borderRadius: '4px',
                        '& .MuiLinearProgress-bar': { backgroundColor: '#00ff88' }
                      }}
                    />
                  </Box>

                  <Typography variant="h6" sx={{ color: '#00ff88', mb: 2 }}>
                    🎯 Next Steps
                  </Typography>
                  <Box component="ul" sx={{ pl: 2, '& li': { mb: 1, color: '#b0b0b0' } }}>
                    <li>Complete requirements gathering</li>
                    <li>Review architecture recommendations</li>
                    <li>Confirm implementation approach</li>
                    <li>Generate production-ready agent</li>
                  </Box>
                </CardContent>
              </Card>

              <Button
                fullWidth
                variant="contained"
                onClick={generateAgent}
                disabled={progress < 80}
                sx={{
                  py: 2,
                  background: 'linear-gradient(135deg, #00ff88 0%, #ff6b6b 100%)',
                  color: '#000000',
                  fontWeight: 'bold',
                  fontSize: '1.1rem'
                }}
              >
                🚀 Generate Agent
              </Button>
            </Grid>
          </Grid>

          {error && (
            <Alert severity="error" sx={{ mt: 2 }}>
              {error}
            </Alert>
          )}
        </Container>
      </Box>
    );
  }

  // Building view
  if (currentView === 'building') {
    return (
      <Box sx={{
        minHeight: '100vh',
        background: 'linear-gradient(135deg, #0a0a0a 0%, #1a1a1a 100%)',
        color: 'white',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center'
      }}>
        <Container maxWidth="md" sx={{ textAlign: 'center' }}>
          <motion.div
            animate={{ rotate: 360 }}
            transition={{ duration: 2, repeat: Infinity, ease: "linear" }}
            style={{ display: 'inline-block', marginBottom: '2rem' }}
          >
            <SparkleIcon sx={{ fontSize: 80, color: '#00ff88' }} />
          </motion.div>

          <Typography variant="h2" gutterBottom>
            Building Your Agent
          </Typography>

          <Typography variant="h4" sx={{ mb: 4, color: '#b0b0b0' }}>
            Generating production-ready code...
          </Typography>

          <LinearProgress
            sx={{
              height: '8px',
              borderRadius: '4px',
              mb: 4,
              '& .MuiLinearProgress-bar': { backgroundColor: '#00ff88' }
            }}
          />

          <Typography variant="body1" sx={{ color: '#b0b0b0' }}>
            This may take a few moments. We're generating:
          </Typography>

          <Box sx={{ mt: 3, display: 'flex', justifyContent: 'center', gap: 2, flexWrap: 'wrap' }}>
            <Chip label="Lambda Functions" sx={{ backgroundColor: '#00ff88', color: '#000000' }} />
            <Chip label="API Gateway" sx={{ backgroundColor: '#00ff88', color: '#000000' }} />
            <Chip label="DynamoDB Tables" sx={{ backgroundColor: '#00ff88', color: '#000000' }} />
            <Chip label="Infrastructure Code" sx={{ backgroundColor: '#00ff88', color: '#000000' }} />
            <Chip label="Documentation" sx={{ backgroundColor: '#00ff88', color: '#000000' }} />
            <Chip label="Deployment Scripts" sx={{ backgroundColor: '#00ff88', color: '#000000' }} />
          </Box>
        </Container>
      </Box>
    );
  }

  // Complete view
  if (currentView === 'complete') {
    return (
      <Box sx={{
        minHeight: '100vh',
        background: 'linear-gradient(135deg, #0a0a0a 0%, #1a1a1a 100%)',
        color: 'white',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center'
      }}>
        <Container maxWidth="md" sx={{ textAlign: 'center' }}>
          <motion.div
            initial={{ scale: 0 }}
            animate={{ scale: 1 }}
            transition={{ duration: 0.5 }}
          >
            <SparkleIcon sx={{ fontSize: 100, color: '#00ff88', mb: 4 }} />
          </motion.div>

          <Typography variant="h1" gutterBottom>
            🎉 Agent Built Successfully!
          </Typography>

          <Typography variant="h3" sx={{ mb: 6, color: '#b0b0b0' }}>
            Your production-ready AI agent is ready for deployment
          </Typography>

          <Grid container spacing={3} sx={{ mb: 6 }}>
            <Grid item xs={12} sm={4}>
              <Card sx={{ textAlign: 'center', py: 3 }}>
                <CardContent>
                  <Typography variant="h2" sx={{ color: '#00ff88', mb: 1 }}>
                    95%+
                  </Typography>
                  <Typography variant="body2">
                    Confidence Score
                  </Typography>
                </CardContent>
              </Card>
            </Grid>

            <Grid item xs={12} sm={4}>
              <Card sx={{ textAlign: 'center', py: 3 }}>
                <CardContent>
                  <Typography variant="h2" sx={{ color: '#00ff88', mb: 1 }}>
                    $16-30
                  </Typography>
                  <Typography variant="body2">
                    Monthly Cost
                  </Typography>
                </CardContent>
              </Card>
            </Grid>

            <Grid item xs={12} sm={4}>
              <Card sx={{ textAlign: 'center', py: 3 }}>
                <CardContent>
                  <Typography variant="h2" sx={{ color: '#00ff88', mb: 1 }}>
                    30-45min
                  </Typography>
                  <Typography variant="body2">
                    Build Time
                  </Typography>
                </CardContent>
              </Card>
            </Grid>
          </Grid>

          <Box sx={{ display: 'flex', gap: 2, justifyContent: 'center', flexWrap: 'wrap' }}>
            <Button
              variant="contained"
              size="large"
              sx={{
                py: 2,
                px: 4,
                background: 'linear-gradient(135deg, #00ff88 0%, #ff6b6b 100%)',
                color: '#000000',
                fontWeight: 'bold'
              }}
            >
              📥 Download Agent Package
            </Button>

            <Button
              variant="outlined"
              size="large"
              onClick={resetToLanding}
              sx={{
                py: 2,
                px: 4,
                borderColor: '#00ff88',
                color: '#00ff88'
              }}
            >
              🔄 Build Another Agent
            </Button>
          </Box>

          <Typography variant="body2" sx={{ mt: 4, color: '#888' }}>
            Your agent includes: Python code, AWS infrastructure, Docker config, documentation, and deployment scripts
          </Typography>
        </Container>
      </Box>
    );
  }

  return null;
}

export default App;
