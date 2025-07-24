import React, { useState, useEffect, useRef, useCallback } from 'react';
import {
  Container,
  Typography,
  Box,
  Button,
  Paper,
  Grid,
  Alert,
  LinearProgress,
  Card,
  CardContent,
  Switch,
  FormControlLabel,
  Chip,
  Tabs,
  Tab
} from '@mui/material';
import StopIcon from '@mui/icons-material/Stop';
import PlayArrowIcon from '@mui/icons-material/PlayArrow';
import PauseIcon from '@mui/icons-material/Pause';
import RestartAltIcon from '@mui/icons-material/RestartAlt';
import Webcam from 'react-webcam';
import { Pose, Results, NormalizedLandmarkList } from '@mediapipe/pose';
import { Camera } from '@mediapipe/camera_utils';
import { drawConnectors, drawLandmarks } from '@mediapipe/drawing_utils';
import { POSE_CONNECTIONS } from '@mediapipe/pose';
import { apiService, PredictionResponse } from '../services/api';
import { getRandomFeedback, getExerciseSpecificFeedback, getMilestoneFeedback } from '../utils/poseDetection';
import { useAuth } from '../contexts/AuthContext';

interface ExerciseMonitorProps {
  selectedExercise: string;
  onBack: () => void;
}

// Function to flatten 33 keypoints (each with x, y, visibility) into a 99-length vector
function flattenLandmarks(landmarks: NormalizedLandmarkList): number[] {
  return landmarks.map(p => [p.x, p.y, p.visibility ?? 0]).flat();
}

const ExerciseMonitor: React.FC<ExerciseMonitorProps> = ({ selectedExercise, onBack }) => {
  const { currentUser } = useAuth();
  const webcamRef = useRef<Webcam>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const poseRef = useRef<Pose | null>(null);
  const cameraRef = useRef<Camera | null>(null);
  
  // Sliding window buffer for keypoint frames
  const frameBufferRef = useRef<number[][]>([]);
  const SEQUENCE_LENGTH = 30;
  
  // Constants for real-time prediction
  const API_URL = "http://localhost:5000/predict"; // Flask server endpoint
  const PREDICT_EVERY = 5; // Send every 5th frame for better performance
  const frameCounterRef = useRef(0);
  
  // Rep detection and feedback state
  const [lastPhase, setLastPhase] = useState<string>('');
  const [repInProgress, setRepInProgress] = useState(false);
  const [lastFeedbackTime, setLastFeedbackTime] = useState(0);
  const [lastRepTime, setLastRepTime] = useState(0);
  const [predictionHistory, setPredictionHistory] = useState<Array<{exercise: string, confidence: number}>>([]);
  const FEEDBACK_COOLDOWN = 1500; // 1.5 seconds between feedback (reduced from 2)
  const REP_COOLDOWN = 1000; // 1 second between rep counting (reduced from 1.5)
  
  // Immediate feedback refs
  const lastSpokenTextRef = useRef<string>('');
  const lastPhaseChangeRef = useRef<number>(0);
  const lastImmediateFeedbackRef = useRef<number>(0);
  
  const [isActive, setIsActive] = useState(false);
  const [isPaused, setIsPaused] = useState(false);
  const [repCount, setRepCount] = useState(0);
  const [currentPhase, setCurrentPhase] = useState<string>('');
  const [confidence, setConfidence] = useState(0);
  const [sessionDuration, setSessionDuration] = useState(0);
  const [prediction, setPrediction] = useState<PredictionResponse | null>(null);
  const [error, setError] = useState('');
  const [voiceEnabled, setVoiceEnabled] = useState(true);
  const [poseDetected, setPoseDetected] = useState(false);
  const [sessionStartTime, setSessionStartTime] = useState<Date | null>(null);
  const [predictedExercise, setPredictedExercise] = useState('');
  const [exerciseFeedback, setExerciseFeedback] = useState('');
  const [formQuality, setFormQuality] = useState<'excellent' | 'good' | 'needs_improvement' | 'poor'>('good');
  const [aiModelDetails, setAiModelDetails] = useState({
    exercise: 'No prediction yet',
    confidence: 0,
    phase: 'waiting',
    rep_count: 0,
    timestamp: '',
    success: false,
    error: ''
  });
  const [consoleLog, setConsoleLog] = useState<string[]>([]);
  const [debugMode, setDebugMode] = useState(false);
  const [showPosePoints, setShowPosePoints] = useState(false);
  const [showLandmarks, setShowLandmarks] = useState(true);
  const [showConnections, setShowConnections] = useState(true);
  const [systemStatus, setSystemStatus] = useState<{
    backend: 'checking' | 'connected' | 'error';
    camera: 'checking' | 'connected' | 'error';
    pose: 'checking' | 'connected' | 'error';
    lastHealthCheck: Date | null;
  }>({
    backend: 'checking',
    camera: 'checking', 
    pose: 'checking',
    lastHealthCheck: null
  });
  const [tabValue, setTabValue] = useState(0);
  const [frameBufferLength, setFrameBufferLength] = useState(0);
  const [keypointsDetected, setKeypointsDetected] = useState(0);

  // Get exercise-specific voice guidance for phase transitions
  const getExerciseSpecificGuidance = (phase: string, exercise: string, isTransition: boolean = false) => {
    const exerciseLower = exercise.toLowerCase();
    
    if (exerciseLower.includes('curl') || exerciseLower.includes('bicep')) {
      if (phase === 'up') {
        return isTransition ? 'Curl up! Bring your hands to your shoulders!' : 'Curling up - good form!';
      } else {
        return isTransition ? 'Lower down slowly! Control the movement!' : 'Lowering - perfect control!';
      }
    } else if (exerciseLower.includes('press') || exerciseLower.includes('bench')) {
      if (phase === 'up') {
        return isTransition ? 'Press up! Push the weight up!' : 'Pressing up - strong push!';
      } else {
        return isTransition ? 'Lower to chest! Control the descent!' : 'Lowering - good control!';
      }
    } else if (exerciseLower.includes('squat')) {
      if (phase === 'up') {
        return isTransition ? 'Stand up! Drive through your heels!' : 'Rising up - great form!';
      } else {
        return isTransition ? 'Squat down! Knees out, chest up!' : 'Squatting - perfect depth!';
      }
    } else if (exerciseLower.includes('pull') || exerciseLower.includes('row')) {
      if (phase === 'up') {
        return isTransition ? 'Pull back! Squeeze your shoulder blades!' : 'Pulling - great squeeze!';
      } else {
        return isTransition ? 'Extend forward! Control the release!' : 'Extending - smooth motion!';
      }
    } else if (exerciseLower.includes('deadlift')) {
      if (phase === 'up') {
        return isTransition ? 'Lift up! Drive with your hips!' : 'Lifting - powerful drive!';
      } else {
        return isTransition ? 'Lower down! Keep your back straight!' : 'Lowering - excellent form!';
      }
    } else {
      // Generic guidance
      if (phase === 'up') {
        return isTransition ? 'Move up! Keep good form!' : 'Up phase - looking good!';
      } else {
        return isTransition ? 'Move down! Control the movement!' : 'Down phase - nice control!';
      }
    }
  };

  // Immediate phase detection from MediaPipe landmarks (no backend delay)
  const detectImmediatePhase = (landmarks: any[], exercise: string) => {
    try {
      // Extract key landmarks for immediate phase detection
      const leftShoulder = landmarks[11];
      const rightShoulder = landmarks[12]; 
      const leftElbow = landmarks[13];
      const rightElbow = landmarks[14];
      const leftWrist = landmarks[15];
      const rightWrist = landmarks[16];
      const leftHip = landmarks[23];
      const rightHip = landmarks[24];
      
      if (!leftElbow || !rightElbow || !leftWrist || !rightWrist || !leftShoulder || !rightShoulder) {
        return null;
      }
      
      // Calculate average positions for stability
      const avgElbowY = (leftElbow.y + rightElbow.y) / 2;
      const avgWristY = (leftWrist.y + rightWrist.y) / 2;
      const avgShoulderY = (leftShoulder.y + rightShoulder.y) / 2;
      const avgHipY = leftHip && rightHip ? (leftHip.y + rightHip.y) / 2 : null;
      
      // Different logic for different exercises
      const exerciseLower = exercise.toLowerCase();
      
      if (exerciseLower.includes('curl') || exerciseLower.includes('bicep')) {
        // For bicep curls: up = hands closer to shoulders (wrists above elbows)
        const armFlexion = avgElbowY - avgWristY;
        return armFlexion > 0.05 ? 'up' : 'down'; // More sensitive threshold
      } else if (exerciseLower.includes('press') || exerciseLower.includes('bench')) {
        // For presses: up = arms extended (wrists above elbows significantly)
        const armExtension = avgElbowY - avgWristY;
        return armExtension > 0.08 ? 'up' : 'down';
      } else if (exerciseLower.includes('squat') && avgHipY) {
        // For squats: up = hips higher, down = hips lower
        const hipHeight = avgHipY;
        return hipHeight < 0.7 ? 'up' : 'down'; // Lower Y value means higher in frame
      } else if (exerciseLower.includes('pull') || exerciseLower.includes('row')) {
        // For pulls: up = elbows behind torso (pulled back)
        const elbowPosition = avgElbowY - avgShoulderY;
        return elbowPosition < -0.02 ? 'up' : 'down';
      } else if (exerciseLower.includes('deadlift') && avgHipY) {
        // For deadlifts: up = standing tall, down = bent over
        const torsoAngle = avgShoulderY - avgHipY;
        return torsoAngle < 0.3 ? 'up' : 'down';
      } else {
        // Default: simple elbow-wrist relationship with better sensitivity
        const armPosition = avgElbowY - avgWristY;
        return armPosition > 0.03 ? 'up' : 'down';
      }
    } catch (error) {
      console.error('Immediate phase detection error:', error);
      return null;
    }
  };

  // Text-to-speech function with immediate feedback optimization
  const speak = (text: string, priority: 'immediate' | 'normal' = 'normal') => {
    console.log('🗣️ SPEAK CALLED:', { text, priority, voiceEnabled, isActive, textLength: text.trim().length });
    
    if (!voiceEnabled) {
      console.log('🚫 Speech blocked: Voice disabled');
      return;
    }
    
    if (!text.trim()) {
      console.log('🚫 Speech blocked: Empty text');
      return;
    }
    
    if (!isActive) {
      console.log('🚫 Speech blocked: Exercise not active - allowing preview speech');
      // Allow speech even when not active for testing purposes
      // return; // Commented out to enable speech in preview mode
    }
    
    try {
      // 🔥 IMMEDIATE CANCELLATION for live feedback
      if (priority === 'immediate') {
        window.speechSynthesis.cancel();
        lastSpokenTextRef.current = ''; // Reset to allow immediate speech
      }
      
      // Don't queue duplicate messages (except for immediate priority)
      if (text === lastSpokenTextRef.current && priority !== 'immediate') {
        console.log(`🚫 Skipping duplicate speech: "${text}"`);
        return;
      }
      
      const utterance = new SpeechSynthesisUtterance(text);
      utterance.rate = priority === 'immediate' ? 1.3 : 1.1; // Faster for immediate feedback
      utterance.pitch = 1.0;
      utterance.volume = 0.9;
      
      // Use fastest available voice for immediate feedback
      const voices = window.speechSynthesis.getVoices();
      const preferredVoice = voices.find(voice => 
        voice.lang.startsWith('en') && voice.localService
      ) || voices.find(voice => voice.lang.startsWith('en'));
      
      if (preferredVoice) {
        utterance.voice = preferredVoice;
      }
      
      // 🚀 IMMEDIATE SPEECH for live feedback
      if (priority === 'immediate') {
        window.speechSynthesis.speak(utterance);
        console.log(`🗣️ IMMEDIATE: "${text}"`);
        lastImmediateFeedbackRef.current = Date.now();
      } else {
        // Only delay for normal priority messages to prevent conflicts
        const delay = Date.now() - lastImmediateFeedbackRef.current < 500 ? 100 : 50;
        setTimeout(() => {
          if (!window.speechSynthesis.speaking) {
            window.speechSynthesis.speak(utterance);
            console.log(`🗣️ Speaking: "${text}"`);
          }
        }, delay);
      }
      
      lastSpokenTextRef.current = text;
    } catch (error) {
      console.error('Speech synthesis error:', error);
    }
  };

  // Helper function to update rep count based on predicted exercise
  const updateRepCount = (predictedExercise: string) => {
    // This is a placeholder for rep counting logic that will be implemented next
    // For now, just log the prediction
    console.log('🎯 Exercise predicted for rep counting:', predictedExercise);
  };

  // Helper function to detect reps based on phase transitions
  const detectRep = (currentPhase: string, exercise: string, confidence: number) => {
    const now = Date.now();
    
    // Enhanced debug logging for BiLSTM rep detection
    console.log("🎯 BiLSTM Rep Detection:", {
      exercise,
      selectedExercise,
      exerciseMatch: exercise.toLowerCase() === selectedExercise.toLowerCase(),
      confidence: confidence.toFixed(3),
      currentPhase,
      lastPhase,
      timeElapsed: now - lastRepTime,
      repCooldown: REP_COOLDOWN
    });
    
    // STRICT requirements for rep counting - must be exact exercise match
    if (exercise.toLowerCase() !== selectedExercise.toLowerCase()) {
      console.log("❌ Rep detection blocked: Wrong exercise", exercise, "vs", selectedExercise);
      addToConsoleLog(`❌ Wrong exercise: Expected ${selectedExercise}, got ${exercise}`);
      return;
    }
    
    // Strict confidence requirement
    if (confidence < 0.75) { // Maintain high confidence threshold
      console.log("❌ Rep detection blocked: Low confidence", confidence);
      addToConsoleLog(`❌ Low confidence: ${(confidence * 100).toFixed(1)}% (need >75%)`);
      return;
    }
    
    if (now - lastRepTime < REP_COOLDOWN) {
      console.log("❌ Rep detection blocked: Cooldown active", now - lastRepTime, "ms");
      return;
    }
    
    // Only count valid phase transitions to prevent false positives
    let repCompleted = false;
    
    // Strict phase transition detection
    if (lastPhase === 'down' && currentPhase === 'up') {
      repCompleted = true;
      console.log("✅ Rep completed: down → up transition for", exercise);
      addToConsoleLog(`🎯 Rep completed: ${lastPhase} → ${currentPhase} (${exercise})`);
    } else if (lastPhase === 'up' && currentPhase === 'hold' && exercise.includes('plank')) {
      repCompleted = true;
      console.log("✅ Plank hold completed: up → hold transition");
      addToConsoleLog(`🎯 Plank hold completed: ${lastPhase} → ${currentPhase}`);
    } else if (lastPhase === 'start' && currentPhase === 'down') {
      // First rep start detection - don't count as completed rep yet
      console.log("🚀 First rep started: start → down for", exercise);
      addToConsoleLog(`🚀 Exercise started: ${lastPhase} → ${currentPhase} (${exercise})`);
      setRepInProgress(true);
    }
    
    if (repCompleted) {
      const newRepCount = repCount + 1;
      setRepCount(newRepCount);
      setLastRepTime(now);
      setRepInProgress(false);
      
      console.log("🎉 REP COUNTED!", {
        newRepCount,
        exercise,
        selectedExercise,
        confidence: confidence.toFixed(3),
        phaseTransition: `${lastPhase} → ${currentPhase}`
      });
      
      addToConsoleLog(`🎉 Rep #${newRepCount} counted for ${exercise}! (${(confidence * 100).toFixed(1)}%)`);
      
      // Immediate UI feedback for rep completion
      setExerciseFeedback(`🎉 Rep ${newRepCount} completed! Great work!`);
      
      // Voice feedback after each rep (immediate, no cooldown for rep counting)
      if (voiceEnabled) {
        let feedback = '';
        
        // Special milestone feedback
        if (newRepCount % 10 === 0) {
          feedback = `Excellent! ${newRepCount} reps completed!`;
        } else if (newRepCount % 5 === 0) {
          feedback = `Great job! ${newRepCount} reps!`;
        } else if (newRepCount <= 3) {
          feedback = `${newRepCount}, keep going!`;
        } else {
          feedback = `${newRepCount}. Good work!`;
        }
        
        speak(feedback, 'immediate'); // Use immediate priority for rep counts
        setLastFeedbackTime(now);
        
        // Also update text feedback
        setExerciseFeedback(feedback);
        addToConsoleLog(`🗣️ Rep feedback: "${feedback}"`);
      }
    }
    
    setLastPhase(currentPhase);
  };

  // Helper function to stabilize predictions using history
  const getStabilizedPrediction = (newExercise: string, newConfidence: number) => {
    // Add to history
    setPredictionHistory(prev => {
      const updated = [...prev.slice(-4), { exercise: newExercise, confidence: newConfidence }];
      return updated;
    });
    
    // Use current prediction history for stabilization
    if (predictionHistory.length < 1) {
      return { exercise: newExercise, confidence: newConfidence, isStable: newConfidence > 0.8 };
    }
    
    // For high confidence predictions, require less history
    const minHistoryRequired = newConfidence > 0.8 ? 1 : 3;
    if (predictionHistory.length < minHistoryRequired) {
      return { exercise: newExercise, confidence: newConfidence, isStable: newConfidence > 0.8 };
    }
    
    // Check if recent predictions are consistent
    const historyToCheck = Math.min(predictionHistory.length, newConfidence > 0.8 ? 2 : 3);
    const recentPredictions = predictionHistory.slice(-historyToCheck);
    const exerciseCounts: { [key: string]: number } = {};
    
    recentPredictions.forEach(pred => {
      exerciseCounts[pred.exercise] = (exerciseCounts[pred.exercise] || 0) + 1;
    });
    
    // Find most common exercise in recent predictions
    const mostCommon = Object.entries(exerciseCounts)
      .sort(([,a], [,b]) => b - a)[0];
    
    // For high confidence, require less agreement
    const requiredAgreement = newConfidence > 0.8 ? 1 : Math.ceil(historyToCheck / 2);
    const isStable = mostCommon && mostCommon[1] >= requiredAgreement;
    const stableExercise = isStable ? mostCommon[0] : newExercise;
    
    // Calculate average confidence for the stable exercise
    const stableConfidence = recentPredictions
      .filter(pred => pred.exercise === stableExercise)
      .reduce((sum, pred) => sum + pred.confidence, 0) / 
      recentPredictions.filter(pred => pred.exercise === stableExercise).length;
    
    return {
      exercise: stableExercise,
      confidence: stableConfidence || newConfidence,
      isStable: isStable || newConfidence > 0.8 // High confidence predictions are immediately stable
    };
  };

  // Timer for session duration
  useEffect(() => {
    let interval: NodeJS.Timeout | null = null;
    
    if (isActive && !isPaused && sessionStartTime) {
      interval = setInterval(() => {
        setSessionDuration(Math.floor((Date.now() - sessionStartTime.getTime()) / 1000));
      }, 1000);
    }
    
    return () => {
      if (interval) clearInterval(interval);
    };
  }, [isActive, isPaused, sessionStartTime]);

  // Initialize MediaPipe Pose
  useEffect(() => {
    const initializePose = async () => {
      try {
        addToConsoleLog('🔄 Initializing MediaPipe Pose...');
        
        // Clear any previous errors
        setError('');
        
        // Add troubleshooting tips
        addToConsoleLog('💡 Tip: Enable "Debug Mode" to see pose landmarks');
        addToConsoleLog('💡 Tip: Ensure good lighting and full body visibility');
        
        // Initialize MediaPipe Pose with proper configuration
        const pose = new Pose({
          locateFile: (file) => `https://cdn.jsdelivr.net/npm/@mediapipe/pose/${file}`
        });

        pose.setOptions({
          modelComplexity: 1,
          smoothLandmarks: true,
          enableSegmentation: false,
          smoothSegmentation: false,
          minDetectionConfidence: 0.5,
          minTrackingConfidence: 0.5
        });

        pose.onResults(onPoseResults);
        poseRef.current = pose;
        console.log('MediaPipe Pose initialized:', poseRef.current);
        
        addToConsoleLog('✅ MediaPipe Pose initialized successfully');
        addToConsoleLog('📱 Ready to start exercise detection!');
        
        // Auto-start camera for pose preview (but not exercise session)
        setTimeout(() => {
          startCameraPreview();
        }, 500);
        
        // Test MediaPipe is working
        try {
          // Create a small test canvas to verify MediaPipe
          const testCanvas = document.createElement('canvas');
          testCanvas.width = 100;
          testCanvas.height = 100;
          const testCtx = testCanvas.getContext('2d');
          if (testCtx) {
            testCtx.fillStyle = 'black';
            testCtx.fillRect(0, 0, 100, 100);
            addToConsoleLog('🎨 Canvas rendering verified');
          }
        } catch (testError) {
          addToConsoleLog('⚠️ Canvas test failed, but continuing...');
        }
        
      } catch (error) {
        console.error('Error initializing pose detection:', error);
        setError('Failed to initialize pose detection. Please refresh the page.');
        addToConsoleLog('❌ MediaPipe initialization failed');
      }
    };

    // Add a small delay to ensure DOM is ready
    // setTimeout(initializePose, 100);
    
    initializePose();

    return () => {
      if (cameraRef.current) {
        cameraRef.current.stop();
      }
    };
  }, []);

  const onPoseResults = useCallback(async (results: Results) => {
    if (!canvasRef.current) return;

    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    // Ensure canvas matches video size
    if (results.image && (canvas.width !== results.image.width || canvas.height !== results.image.height)) {
      canvas.width = results.image.width;
      canvas.height = results.image.height;
    }

    // Clear canvas
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    // Always draw the video frame first
    if (results.image) {
      ctx.drawImage(results.image, 0, 0, canvas.width, canvas.height);
    }

    if (results.poseLandmarks && results.poseLandmarks.length > 0) {
      if (!poseDetected) {
        addToConsoleLog('👤 Pose detected! Processing keypoints...');
      }
      setPoseDetected(true);
      setKeypointsDetected(results.poseLandmarks.length);
      
      // 🧮 STEP 1: Flatten the 33 keypoints into a 99-length vector
      const currentFrame = flattenLandmarks(results.poseLandmarks);
      
      // 🔥 IMMEDIATE PHASE DETECTION - Don't wait for backend!
      if (selectedExercise && results.poseLandmarks.length === 33) {
        const immediatePhase = detectImmediatePhase(results.poseLandmarks, selectedExercise);
        const now = Date.now();
        
        // Check landmark visibility for form feedback
        const avgVisibility = results.poseLandmarks.reduce((sum, landmark) => sum + (landmark.visibility || 0), 0) / 33;
        
        if (immediatePhase && immediatePhase !== currentPhase && now - lastPhaseChangeRef.current > 300) {
          console.log(`🚀 IMMEDIATE phase change: ${currentPhase} → ${immediatePhase}`);
          
          // ❌ DISABLED: Immediate rep counting to avoid conflicts with backend BiLSTM
          // Backend provides more accurate rep counting through BiLSTM analysis
          /* DISABLED - Let backend handle rep counting
          if (lastPhase && lastPhase !== immediatePhase) {
            // Count a rep when transitioning from down to up (completing the concentric phase)
            if (lastPhase === 'down' && immediatePhase === 'up' && now - lastRepTime > REP_COOLDOWN) {
              const newRepCount = repCount + 1;
              setRepCount(newRepCount);
              setLastRepTime(now);
              setRepInProgress(false);
              
              console.log(`🎉 IMMEDIATE REP COUNTED: ${newRepCount}`);
              addToConsoleLog(`🎉 Rep #${newRepCount} completed! (Immediate detection)`);
              
              // Immediate rep completion feedback
              if (voiceEnabled) {
                let repFeedback = '';
                if (newRepCount % 10 === 0) {
                  repFeedback = `Outstanding! ${newRepCount} reps completed!`;
                } else if (newRepCount % 5 === 0) {
                  repFeedback = `Excellent! ${newRepCount} reps done!`;
                } else {
                  repFeedback = `${newRepCount}! Keep going!`;
                }
                speak(repFeedback, 'immediate');
                setExerciseFeedback(`🎉 Rep ${newRepCount} completed!`);
              }
            } else if (lastPhase === 'up' && immediatePhase === 'down') {
              // Starting a new rep (eccentric phase)
              setRepInProgress(true);
              console.log(`🔄 Starting new rep: ${lastPhase} → ${immediatePhase}`);
            }
          }
          */
          
          // Track rep progress for UI indication
          if (lastPhase && lastPhase !== immediatePhase) {
            if (lastPhase === 'up' && immediatePhase === 'down') {
              setRepInProgress(true);
              console.log(`🔄 Rep in progress: ${lastPhase} → ${immediatePhase}`);
            } else if (lastPhase === 'down' && immediatePhase === 'up') {
              setRepInProgress(false);
              console.log(`🔄 Rep phase completed: ${lastPhase} → ${immediatePhase}`);
            }
          }
          
          setCurrentPhase(immediatePhase);
          setLastPhase(currentPhase); // Update lastPhase for next transition
          lastPhaseChangeRef.current = now;
          
          // 🗣️ EXERCISE-SPECIFIC LIVE VOICE GUIDANCE (only during active exercise)
          if (voiceEnabled && isActive && now - lastImmediateFeedbackRef.current > 800) { // Increased to prevent speech overlap
            let feedback = '';
            
            // Get exercise-specific guidance based on phase transition
            if (avgVisibility > 0.7) {
              // Good visibility - give specific exercise guidance
              feedback = getExerciseSpecificGuidance(immediatePhase, selectedExercise, true);
            } else if (avgVisibility > 0.5) {
              // Medium visibility - basic guidance
              feedback = getExerciseSpecificGuidance(immediatePhase, selectedExercise, false);
            } else {
              // Poor visibility - positioning feedback
              feedback = 'Move into better position for detection';
            }
            
            speak(feedback, 'immediate');
            setExerciseFeedback(feedback);
          }
        }
      }
      
      // 🧠 STEP 2: Maintain sliding window buffer of last 30 frames
      frameBufferRef.current.push(currentFrame);
      if (frameBufferRef.current.length > SEQUENCE_LENGTH) {
        frameBufferRef.current.shift(); // Remove oldest frame
      }
      
      // Update buffer length state for UI display
      setFrameBufferLength(frameBufferRef.current.length);
      
      // Add buffer status to console log
      if (frameBufferRef.current.length === 1 || frameBufferRef.current.length % 10 === 0) {
        addToConsoleLog(`📊 Buffer: ${frameBufferRef.current.length}/${SEQUENCE_LENGTH} frames, Vector: ${currentFrame.length} values`);
      }
      
      // 🚀 STEP 3: Send buffer to backend for real-time prediction when ready
      frameCounterRef.current++;
      
      // 🔥 ENHANCED DEBUGGING for BiLSTM pipeline
      console.log("🔍 BiLSTM Pipeline Check:", {
        bufferLength: frameBufferRef.current.length,
        requiredLength: SEQUENCE_LENGTH,
        frameCounter: frameCounterRef.current,
        predictEvery: PREDICT_EVERY,
        shouldPredict: frameBufferRef.current.length === SEQUENCE_LENGTH && frameCounterRef.current % PREDICT_EVERY === 0,
        isActive,
        isPaused
      });
      
      // � STEP 3: Send buffer to backend for real-time prediction when ready
      if (frameBufferRef.current.length === SEQUENCE_LENGTH && frameCounterRef.current % PREDICT_EVERY === 0) {
        // Safety check before sending
        if (frameBufferRef.current.some(f => f.length !== 99)) {
          console.warn("⚠️ Frame buffer contains invalid data");
          addToConsoleLog("⚠️ Frame buffer contains invalid data, skipping prediction");
          return;
        }
        
        // Use the full 30-frame sequence
        const sequenceToSend = [...frameBufferRef.current];
        
        // Send frame buffer to backend for BiLSTM prediction
        console.log('🚀 SENDING PREDICTION REQUEST:', {
          url: API_URL,
          sequenceShape: [sequenceToSend.length, sequenceToSend[0]?.length],
          originalFrames: frameBufferRef.current.length,
          paddedFrames: sequenceToSend.length,
          selectedExercise,
          timestamp: new Date().toISOString()
        });
        addToConsoleLog(`🚀 Sending ${sequenceToSend.length}-frame sequence to BiLSTM model...`);
        
        fetch(API_URL, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ 
            sequence: sequenceToSend,
            selected_exercise: selectedExercise // Include selected exercise for validation
          }),
        })
          .then(res => {
            if (!res.ok) {
              throw new Error(`HTTP ${res.status}: ${res.statusText}`);
            }
            return res.json();
          })
          .then(data => {
            const { exercise, confidence, phase, rep_count, debug_info } = data;
            
            // Enhanced BiLSTM debugging
            console.log("🤖 BiLSTM Raw Response:", {
              exercise,
              confidence: confidence.toFixed(3),
              phase,
              rep_count,
              backend_rep_count: rep_count,
              frontend_rep_count: repCount,
              selected: selectedExercise,
              debug_info,
              timestamp: new Date().toISOString()
            });
            
            // Get stabilized prediction to reduce flickering
            const stabilized = getStabilizedPrediction(exercise, confidence);
            
            console.log("🧠 Prediction Stability:", {
              rawExercise: exercise,
              stabilizedExercise: stabilized.exercise,
              rawConfidence: confidence.toFixed(3),
              stabilizedConfidence: stabilized.confidence.toFixed(3),
              isStable: stabilized.isStable,
              historyLength: predictionHistory.length
            });
            
            // Only update UI if prediction is stable and matches selected exercise
            const shouldUpdate = stabilized.isStable;
            const isCorrectExercise = stabilized.exercise.toLowerCase() === selectedExercise.toLowerCase();
            
            // ✅ ENABLE: Update rep count from backend ONLY for correct exercise
            // Backend BiLSTM model provides more accurate rep counting than immediate detection
            if (rep_count !== undefined && rep_count > repCount && isCorrectExercise && stabilized.confidence > 0.75) {
              console.log(`📊 Backend rep count update: ${repCount} → ${rep_count} (Correct exercise: ${stabilized.exercise})`);
              addToConsoleLog(`🎉 Backend counted rep #${rep_count} for ${stabilized.exercise}!`);
              setRepCount(rep_count); // Update to backend count
              setLastRepTime(Date.now());
              
              // Voice feedback for new rep from backend
              if (voiceEnabled) {
                let repFeedback = '';
                if (rep_count % 10 === 0) {
                  repFeedback = `Outstanding! ${rep_count} reps completed!`;
                } else if (rep_count % 5 === 0) {
                  repFeedback = `Excellent! ${rep_count} reps done!`;
                } else {
                  repFeedback = `${rep_count}! Keep going!`;
                }
                speak(repFeedback, 'immediate');
                setExerciseFeedback(`🎉 Rep ${rep_count} completed!`);
              }
            } else if (rep_count !== undefined && rep_count > repCount && !isCorrectExercise && stabilized.confidence > 0.70) {
              // Wrong exercise detected - don't count reps, give corrective feedback
              console.log(`❌ Wrong exercise detected: Expected ${selectedExercise}, got ${stabilized.exercise} - NOT counting rep`);
              addToConsoleLog(`❌ Wrong exercise: Expected ${selectedExercise}, detected ${stabilized.exercise}`);
              
              // Provide corrective feedback
              const correctExerciseName = selectedExercise.replace(/_/g, ' ').toLowerCase();
              const detectedExerciseName = stabilized.exercise.replace(/_/g, ' ').toLowerCase();
              const correctionMessage = `Please perform ${correctExerciseName}. I'm detecting ${detectedExerciseName} instead.`;
              
              setExerciseFeedback(correctionMessage);
              setFormQuality('poor');
              
              // Voice correction feedback (but not too frequently)
              const now = Date.now();
              if (voiceEnabled && now - lastFeedbackTime > 3000) { // 3 seconds between correction feedback
                speak(correctionMessage, 'immediate');
                setLastFeedbackTime(now);
              }
            } else if (rep_count !== undefined && rep_count !== repCount) {
              console.log(`📊 Backend rep count: ${rep_count}, Frontend rep count: ${repCount}, Exercise match: ${isCorrectExercise}`);
              addToConsoleLog(`📊 Backend suggests ${rep_count} reps, frontend has ${repCount} (${isCorrectExercise ? 'correct' : 'wrong'} exercise)`);
            }
            
            if (shouldUpdate) {
              // Only log to console, don't show exercise classification to user during exercise
              addToConsoleLog(`🤖 AI Detection: ${exercise} (${(confidence * 100).toFixed(1)}%)`);
              
              // Update confidence but don't show predicted exercise to user
              setConfidence(confidence || 0);
              
              // Only set predicted exercise if it matches selected exercise
              if (isCorrectExercise) {
                setPredictedExercise(exercise || 'Unknown');
              }
              
              // Update phase and detect reps - only for correct exercise with high confidence
              if (phase && phase !== currentPhase && isCorrectExercise && stabilized.confidence > 0.75) {
                console.log("📈 Phase transition detected for CORRECT exercise:", currentPhase, "→", phase, `(${stabilized.exercise})`);
                setCurrentPhase(phase);
                detectRep(phase, stabilized.exercise, stabilized.confidence);
              } else if (phase && !isCorrectExercise && stabilized.confidence > 0.70) {
                console.log("❌ Ignoring phase transition - wrong exercise detected:", stabilized.exercise, "instead of", selectedExercise);
                addToConsoleLog(`❌ Wrong exercise: detected ${stabilized.exercise}, expected ${selectedExercise}`);
                // Update phase for display but don't count reps
                setCurrentPhase(phase);
              } else if (phase && isCorrectExercise && stabilized.confidence <= 0.75) {
                console.log("⚠️ Correct exercise but low confidence - not counting reps:", stabilized.confidence);
                addToConsoleLog(`⚠️ Low confidence ${stabilized.exercise}: ${(stabilized.confidence * 100).toFixed(1)}%`);
                setCurrentPhase(phase);
              }
              
              // Update AI model details for UI display (backend info only, not shown to user during exercise)
              setAiModelDetails({
                exercise: isCorrectExercise ? stabilized.exercise : 'Wrong Exercise Detected',
                confidence: stabilized.confidence,
                phase: isCorrectExercise ? (phase || 'unknown') : 'ignored',
                rep_count: rep_count || 0,
                timestamp: new Date().toISOString(),
                success: true,
                error: ''
              });
              
              // Generate form feedback after each rep completion (not continuously)
              const now = Date.now();
              
              // Only give form feedback after rep completion OR if there's a wrong exercise detected
              if ((rep_count > repCount && isCorrectExercise) || (!isCorrectExercise && stabilized.confidence > 0.7)) {
                let feedback = '';
                let shouldSpeak = false;
                
                if (!isCorrectExercise && stabilized.confidence > 0.7) {
                  // Wrong exercise detected with high confidence
                  setFormQuality('poor');
                  const correctExerciseName = selectedExercise.replace(/_/g, ' ').toLowerCase();
                  const detectedExerciseName = stabilized.exercise.replace(/_/g, ' ').toLowerCase();
                  feedback = `Wrong exercise! Please perform ${correctExerciseName}. I detected ${detectedExerciseName}.`;
                  shouldSpeak = true;
                  console.log(`❌ Exercise mismatch: Expected ${correctExerciseName}, detected ${detectedExerciseName}`);
                } else if (isCorrectExercise && rep_count > repCount) {
                  // Form feedback after completing a rep of the CORRECT exercise
                  if (stabilized.confidence >= 0.85) {
                    setFormQuality('excellent');
                    feedback = "Perfect form on that rep!";
                  } else if (stabilized.confidence >= 0.70) {
                    setFormQuality('good');
                    feedback = "Good rep! Keep that form.";
                  } else if (stabilized.confidence >= 0.50) {
                    setFormQuality('needs_improvement');
                    feedback = "Rep completed. Focus on technique next time.";
                  } else {
                    setFormQuality('poor');
                    feedback = "Rep counted, but work on your form.";
                  }
                  shouldSpeak = true;
                }
                
                if (shouldSpeak && feedback) {
                  setExerciseFeedback(feedback);
                  // Different cooldowns for different types of feedback
                  const feedbackCooldown = !isCorrectExercise ? 4000 : 2000; // Longer cooldown for correction feedback
                  if (now - lastFeedbackTime > feedbackCooldown) {
                    speak(feedback, 'normal'); // Use normal priority for form feedback
                    setLastFeedbackTime(now);
                  }
                }
              }
              
            } else {
              // Just log unstable predictions without updating UI
              console.log("🔄 Prediction unstable, waiting for consistency...");
              addToConsoleLog(`🔄 Stabilizing... (${predictionHistory.length}/3 predictions)`);
            }
          })
          .catch(err => {
            console.error("❌ Prediction error:", err);
            addToConsoleLog(`❌ Prediction failed: ${err.message}`);
            
            // Update AI model details with error state
            setAiModelDetails({
              exercise: 'Error',
              confidence: 0,
              phase: 'error',
              rep_count: 0,
              timestamp: new Date().toISOString(),
              success: false,
              error: err.message
            });
          });
      }
      
      // Draw pose landmarks and connections based on toggles
      try {
        if (showConnections) {
          drawConnectors(ctx, results.poseLandmarks, POSE_CONNECTIONS, {
            color: '#00FF00',
            lineWidth: 3
          });
        }
        if (showLandmarks) {
          drawLandmarks(ctx, results.poseLandmarks, {
            color: '#FF0000',
            lineWidth: 2,
            radius: 4
          });
        }
        
        // Add pose detection indicator
        ctx.fillStyle = '#00FF00';
        ctx.font = '16px Arial';
        ctx.fillText('✅ POSE DETECTED', 10, 30);
        
        // Add buffer status indicator
        ctx.fillStyle = '#00FFFF';
        ctx.font = '14px Arial';
        ctx.fillText(`Buffer: ${frameBufferRef.current.length}/${SEQUENCE_LENGTH}`, 10, 50);
        
        // Add debug info if debug mode is enabled
        if (debugMode) {
          ctx.fillStyle = '#FFFF00';
          ctx.font = '12px Arial';
          ctx.fillText(`Landmarks: ${results.poseLandmarks.length}`, 10, 70);
          ctx.fillText(`Frame Vector: ${currentFrame.length}`, 10, 85);
          ctx.fillText(`Timestamp: ${Date.now()}`, 10, 100);
          
          // Draw landmark indices for debugging
          results.poseLandmarks.forEach((landmark: any, index: number) => {
            if (index % 5 === 0) { // Show every 5th landmark to avoid clutter
              const x = landmark.x * canvas.width;
              const y = landmark.y * canvas.height;
              ctx.fillStyle = '#YELLOW';
              ctx.font = '10px Arial';
              ctx.fillText(index.toString(), x + 5, y - 5);
            }
          });
        }
      } catch (drawError) {
        console.error('Drawing error:', drawError);
        addToConsoleLog(`❌ Drawing error: ${drawError}`);
      }
      // TODO: In the next step, we'll send the buffer to backend when it reaches 30 frames
      // For now, we just maintain the buffer and extract joint angles for existing functionality
      
      // ❌ JOINT ANGLE PIPELINE DISABLED - Using BiLSTM keypoint system only
      // Extract joint angles for existing functionality (backward compatibility) - COMMENTED OUT
      /* DISABLED - Joint angle system conflicts with BiLSTM pipeline
      try {
        const jointAngles = extractJointAngles(results.poseLandmarks);
        
        // Debug logging if debug mode is enabled
        if (debugMode) {
          addToConsoleLog(`🔧 Debug: Joint angles: [${jointAngles.map(a => a.toFixed(1)).join(', ')}]`);
        }
        
        // Validate joint angles before sending
        if (!jointAngles || jointAngles.length !== 9) {
          addToConsoleLog('⚠️ Invalid joint angles extracted');
          return;
        }
        
        // Check for reasonable angle values (0-180 degrees)
        const invalidAngles = jointAngles.filter(angle => angle < 0 || angle > 180 || isNaN(angle));
        if (invalidAngles.length > 0) {
          addToConsoleLog(`⚠️ Detected ${invalidAngles.length} invalid angle(s), skipping prediction`);
          return;
        }
        
        // Make prediction with retry logic
        let predictionResult;
        try {
          predictionResult = await apiService.predictExercise(jointAngles, selectedExercise);
        } catch (apiError) {
          addToConsoleLog(`❌ API prediction failed: ${apiError}`);
          return;
        }
        
        // Validate prediction result
        if (!predictionResult || typeof predictionResult.confidence !== 'number') {
          addToConsoleLog('❌ Invalid prediction result received');
          return;
        }
        
        // Only log significant changes to avoid spam
        if (Math.abs(predictionResult.confidence - confidence) > 0.1 || 
            predictionResult.exercise !== predictedExercise) {
          addToConsoleLog(`🤖 BiLSTM: ${predictionResult.exercise} (${(predictionResult.confidence * 100).toFixed(1)}%)`);
        }
        
        // Update states
        setPrediction(predictionResult);
        setRepCount(predictionResult.rep_count);
        setCurrentPhase(predictionResult.phase);
        setConfidence(predictionResult.confidence);
        
        // Set AI model detailed outputs
        setPredictedExercise(predictionResult.exercise || 'Unknown');
        setAiModelDetails({
          exercise: predictionResult.exercise || 'Unknown',
          confidence: predictionResult.confidence || 0,
          phase: predictionResult.phase || 'unknown',
          rep_count: predictionResult.rep_count || 0,
          timestamp: new Date().toISOString(),
          success: true,
          error: ''
        });
        
        // Generate exercise feedback based on confidence and form
        const generateFeedback = (result: PredictionResponse) => {
          const confidence = result.confidence;
          const isCorrectExercise = result.exercise?.toLowerCase() === selectedExercise.toLowerCase();
          
          if (confidence >= 0.85 && isCorrectExercise) {
            setFormQuality('excellent');
            return "Perfect form! Keep it up!";
          } else if (confidence >= 0.70 && isCorrectExercise) {
            setFormQuality('good');
            return "Good form! Stay focused on your movement.";
          } else if (confidence >= 0.50) {
            setFormQuality('needs_improvement');
            return isCorrectExercise 
              ? "Form needs improvement. Focus on proper technique."
              : `AI detected ${result.exercise?.replace(/_/g, ' ')} instead of ${selectedExercise.replace(/_/g, ' ')}`;
          } else {
            setFormQuality('poor');
            return "Low confidence detection. Check your positioning.";
          }
        };
        
        const newFeedback = generateFeedback(predictionResult);
        setExerciseFeedback(newFeedback);
        
        // Log rep count changes and feedback
        if (predictionResult.rep_count > repCount) {
          addToConsoleLog(`🔢 Rep count: ${repCount} → ${predictionResult.rep_count}`);
        }
        if (predictionResult.phase !== currentPhase) {
          addToConsoleLog(`📈 Phase: ${currentPhase} → ${predictionResult.phase}`);
        }

        // Voice feedback based on rep count and confidence
        if (voiceEnabled && predictionResult.rep_count > repCount) {
          let feedback: string;
          if (predictionResult.rep_count % 5 === 0 || predictionResult.rep_count <= 3) {
            feedback = getMilestoneFeedback(predictionResult.rep_count);
          } else {
            feedback = Math.random() > 0.5 
              ? getRandomFeedback('goodRep')
              : getExerciseSpecificFeedback(selectedExercise);
          }
          speak(feedback);
        }

      } catch (error) {
        console.error('Error processing pose results:', error);
        addToConsoleLog(`❌ Pose processing error: ${error}`);
      }
      */ 
      // END DISABLED JOINT ANGLE SYSTEM
    } else {
      console.log('❌ No pose landmarks detected');
      if (poseDetected) {
        addToConsoleLog('❌ Pose lost');
      }
      setPoseDetected(false);
      setKeypointsDetected(0);
      
      // Draw "No Pose" indicator
      ctx.fillStyle = '#FF0000';
      ctx.font = '16px Arial';
      ctx.fillText('❌ NO POSE DETECTED', 10, 30);
    }
  }, [isActive, isPaused, repCount, currentPhase, voiceEnabled, selectedExercise, poseDetected, confidence, predictedExercise, debugMode, showLandmarks, showConnections]);

  const startCameraPreview = useCallback(async () => {
    try {
      console.log('📹 Starting camera preview...');
      addToConsoleLog('🎥 Starting camera preview...');

      if (!poseRef.current) {
        addToConsoleLog('❌ MediaPipe Pose not ready');
        return;
      }

      if (!webcamRef.current?.video) {
        addToConsoleLog('❌ Webcam video element not found');
        return;
      }

      // Request camera permissions
      const stream = await navigator.mediaDevices.getUserMedia({ 
        video: { 
          width: 640, 
          height: 480,
          facingMode: 'user'
        } 
      });

      // Assign stream to webcam
      if (webcamRef.current.video.srcObject !== stream) {
        webcamRef.current.video.srcObject = stream;
      }

      const video = webcamRef.current.video;

      // Wait for video to be ready
      await new Promise(resolve => {
        if (video.readyState >= 3) {
          resolve(true);
        } else {
          video.onloadeddata = () => resolve(true);
        }
      });

      // Force video to play
      if (video.paused) {
        await video.play();
      }

      // Update canvas size
      if (canvasRef.current) {
        canvasRef.current.width = video.videoWidth || 640;
        canvasRef.current.height = video.videoHeight || 480;
        addToConsoleLog(`🎨 Canvas sized to: ${canvasRef.current.width}x${canvasRef.current.height}`);
      }

      // Create MediaPipe Camera for preview
      const camera = new Camera(video, {
        onFrame: async () => {
          // Send frame to MediaPipe for pose detection (preview mode)
          if (video && poseRef.current) {
            try {
              await poseRef.current.send({ image: video });
            } catch (error) {
              console.error('❌ Error in preview mode:', error);
            }
          }
        },
        width: 640,
        height: 480
      });
      cameraRef.current = camera;
      await camera.start();
      addToConsoleLog('✅ Camera preview started!');
      
    } catch (error) {
      console.error('Camera preview error:', error);
      addToConsoleLog(`❌ Camera preview error: ${error}`);
    }
  }, []);

  const startCamera = useCallback(async () => {
    try {
      console.log('📹 Starting camera...');
      addToConsoleLog('🎥 Starting camera...');
      setError('');

      if (!poseRef.current) {
        addToConsoleLog('❌ MediaPipe Pose not initialized');
        setError('Pose detection not ready. Please refresh the page.');
        return;
      }

      if (!webcamRef.current?.video) {
        addToConsoleLog('❌ Webcam video element not found');
        setError('Camera not available. Please allow camera permissions.');
        return;
      }

      // Request camera permissions explicitly (like in MediaPipeDebug)
      const stream = await navigator.mediaDevices.getUserMedia({ 
        video: { 
          width: 640, 
          height: 480,
          facingMode: 'user'
        } 
      });
      console.log('✅ Camera permissions granted');

      // Assign stream to webcam video element if not already set
      if (webcamRef.current.video.srcObject !== stream) {
        webcamRef.current.video.srcObject = stream;
      }

      const video = webcamRef.current.video;
      console.log('Camera started, video element:', video);

      // Wait for video to be ready
      await new Promise(resolve => {
        if (video.readyState >= 3) {
          resolve(true);
        } else {
          video.onloadeddata = () => resolve(true);
        }
      });

      // Force video to play
      if (video.paused) {
        await video.play();
        console.log('Forced video to play');
      }

      // Update canvas size to match video
      if (canvasRef.current) {
        canvasRef.current.width = video.videoWidth || 640;
        canvasRef.current.height = video.videoHeight || 480;
        addToConsoleLog(`🎨 Canvas sized to: ${canvasRef.current.width}x${canvasRef.current.height}`);
      }

      // Set isActive to true BEFORE starting camera
      setIsActive(true);

      // Create MediaPipe Camera with proper configuration
      const camera = new Camera(video, {
        onFrame: async () => {
          // Send frame to MediaPipe for pose detection
          if (video && poseRef.current) {
            try {
              await poseRef.current.send({ image: video });
            } catch (error) {
              console.error('❌ Error sending frame to pose detection:', error);
            }
          }
        },
        width: 640,
        height: 480
      });
      cameraRef.current = camera;
      await camera.start();
      addToConsoleLog('✅ Camera started successfully!');
      
    } catch (error) {
      console.error('Camera start error:', error);
      addToConsoleLog(`❌ Camera error: ${error}`);
      setError('Failed to start camera. Please check permissions and refresh.');
    }
  }, [isActive, isPaused]);

  const handleStart = async () => {
    try {
      setError('');
      addToConsoleLog('🚀 Starting exercise session...');
      
      // Pre-flight checks - like in the working test
      addToConsoleLog('🔍 Running pre-flight checks...');
      
      // Check backend connectivity
      try {
        const health = await apiService.healthCheck();
        addToConsoleLog(`✅ Backend healthy: ${health.status}`);
      } catch (backendError) {
        addToConsoleLog('❌ Backend connection failed');
        setError('Backend connection failed. Please ensure the backend is running.');
        return;
      }
      
      // Check if selected exercise is available
      try {
        const exercises = await apiService.getExercises();
        if (!exercises.includes(selectedExercise)) {
          addToConsoleLog(`❌ Exercise "${selectedExercise}" not found in backend`);
          setError(`Exercise "${selectedExercise}" not available in backend.`);
          return;
        }
        addToConsoleLog(`✅ Exercise "${selectedExercise}" validated`);
      } catch (exerciseError) {
        addToConsoleLog('❌ Failed to validate exercise');
        setError('Failed to validate exercise with backend.');
        return;
      }
      
      // Reset session
      await apiService.resetSession();
      addToConsoleLog('✅ Session reset successful');
      
      // Reset frame buffer for new session
      frameBufferRef.current = [];
      frameCounterRef.current = 0;
      setFrameBufferLength(0);
      setKeypointsDetected(0);
      setPredictionHistory([]);
      setLastPhase('');
      setRepInProgress(false);
      setLastFeedbackTime(0);
      setLastRepTime(0);
      addToConsoleLog('🔄 Frame buffer and rep detection reset for new session');
      
      // Initialize session state
      setIsActive(true);
      setIsPaused(false);
      setRepCount(0);
      setSessionDuration(0);
      setSessionStartTime(new Date());
      setPredictedExercise('');
      setExerciseFeedback('Ready to start!');
      
      // Camera should already be running from preview, just ensure it's active
      if (!cameraRef.current) {
        addToConsoleLog('🎥 Starting camera for exercise session...');
        await startCamera();
      } else {
        addToConsoleLog('🎥 Using existing camera preview for exercise session');
      }
      
      if (voiceEnabled) {
        speak(`Starting ${selectedExercise.replace(/_/g, ' ')} exercise. Good luck!`);
      }
      
      addToConsoleLog('🎯 Exercise session started successfully!');
      
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : 'Unknown error';
      addToConsoleLog(`❌ Start failed: ${errorMessage}`);
      setError(`Failed to start exercise session: ${errorMessage}`);
      console.error('Start error:', error);
    }
  };

  const handlePause = () => {
    setIsPaused(!isPaused);
    
    // Stop any ongoing speech when pausing
    if (!isPaused) {
      window.speechSynthesis.cancel();
      lastSpokenTextRef.current = '';
    }
    
    if (voiceEnabled) {
      speak(isPaused ? 'Resuming exercise' : 'Exercise paused');
    }
  };

  const handleStop = async () => {
    try {
      setIsActive(false);
      setIsPaused(false);
      
      // 🔥 IMMEDIATELY STOP ALL VOICE FEEDBACK
      window.speechSynthesis.cancel();
      lastSpokenTextRef.current = '';
      
      if (cameraRef.current) {
        cameraRef.current.stop();
      }

      // Prepare comprehensive session data
      const sessionData = {
        user_id: currentUser?.uid || 'anonymous',
        exercise: selectedExercise,
        total_reps: repCount,
        duration: sessionDuration,
        confidence_avg: confidence,
        form_quality: formQuality,
        session_data: [{
          exercise: selectedExercise,
          reps: repCount,
          duration: sessionDuration,
          confidence: confidence,
          form_quality: formQuality,
          timestamp: new Date().toISOString()
        }]
      };

      // Log session to backend with error handling
      try {
        if (currentUser && sessionStartTime) {
          const response = await apiService.logSession(sessionData);
          console.log('✅ Session logged successfully:', response);
          addToConsoleLog(`✅ Session saved: ${repCount} reps, ${formatTime(sessionDuration)}`);
        } else {
          console.log('⚠️ Session not logged - no user or start time');
          addToConsoleLog('⚠️ Session not saved - user authentication required');
        }
      } catch (logError) {
        console.error('❌ Failed to log session:', logError);
        addToConsoleLog(`❌ Failed to save session: ${logError}`);
      }

      // Comprehensive completion feedback
      const completionMessage = `Exercise completed! You did ${repCount} repetitions in ${formatTime(sessionDuration)}.`;
      setExerciseFeedback(completionMessage);
      
      if (voiceEnabled) {
        speak(completionMessage);
      }
      
      // Reset session state
      setSessionStartTime(null);
      
      addToConsoleLog(`🏁 Exercise completed: ${repCount} reps, ${formatTime(sessionDuration)}, ${(confidence * 100).toFixed(1)}% avg confidence`);
      
    } catch (error) {
      console.error('Error stopping session:', error);
      addToConsoleLog(`❌ Error stopping session: ${error}`);
    }
  };

  const formatTime = (seconds: number): string => {
    const mins = Math.floor(seconds / 60);
    const secs = seconds % 60;
    return `${mins.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
  };

  const getPhaseColor = (phase: string) => {
    switch (phase.toLowerCase()) {
      case 'up':
        return 'success';
      case 'down':
        return 'info';
      default:
        return 'default';
    }
  };

  const getFormQualityColor = (quality: string) => {
    switch (quality) {
      case 'excellent':
        return 'success';
      case 'good':
        return 'info';
      case 'needs_improvement':
        return 'warning';
      case 'poor':
        return 'error';
      default:
        return 'default';
    }
  };

  const getFormQualityIcon = (quality: string) => {
    switch (quality) {
      case 'excellent':
        return '🏆';
      case 'good':
        return '✅';
      case 'needs_improvement':
        return '⚠️';
      case 'poor':
        return '❌';
      default:
        return '🔄';
    }
  };

  const addToConsoleLog = (message: string) => {
    const timestamp = new Date().toLocaleTimeString();
    const logEntry = `[${timestamp}] ${message}`;
    setConsoleLog(prev => [...prev.slice(-9), logEntry]); // Keep last 10 entries
  };

  const testConnection = async () => {
    try {
      addToConsoleLog('🧪 Running connection test...');
      
      // Test backend health
      const health = await apiService.healthCheck();
      addToConsoleLog(`✅ Backend test: ${health.status}`);
      
      // Test exercise list
      const exercises = await apiService.getExercises();
      addToConsoleLog(`✅ Found ${exercises.length} exercises`);
      
      // ✅ Test BiLSTM prediction endpoint (not joint angles)
      try {
        // Test with dummy 30-frame sequence
        const dummyFrame = Array(99).fill(0.5); // 99 keypoint values
        const dummySequence = Array(30).fill(dummyFrame); // 30 frames
        
        const response = await fetch(API_URL, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ 
            sequence: dummySequence,
            selected_exercise: selectedExercise
          }),
        });
        
        if (!response.ok) {
          throw new Error(`HTTP ${response.status}: ${response.statusText}`);
        }
        
        const testResult = await response.json();
        addToConsoleLog(`✅ BiLSTM test: ${testResult.exercise} (${(testResult.confidence * 100).toFixed(1)}%)`);
      } catch (error) {
        addToConsoleLog(`❌ BiLSTM test failed: ${error}`);
      }
      
      addToConsoleLog('🎉 All tests completed!');
      
    } catch (error) {
      addToConsoleLog(`❌ Connection test failed: ${error}`);
    }
  };

  return (
    <Container maxWidth="xl" sx={{ mt: 2 }}>
      <Box sx={{ mb: 3, display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <Typography variant="h4" component="h1">
          {selectedExercise.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase())} Monitor
        </Typography>
        <Button variant="outlined" onClick={onBack}>
          Back to Exercises
        </Button>
      </Box>

      {error && (
        <Alert severity="error" sx={{ mb: 2 }}>
          {error}
        </Alert>
      )}

      <Grid container spacing={2}>
        {/* Camera Feed */}
        <Grid item xs={12} md={7}>
          <Paper sx={{ p: 1, position: 'relative', height: '100%' }}>
            <Box sx={{ position: 'relative', width: '100%', aspectRatio: '4/3' }}>
              <Webcam
                ref={webcamRef}
                style={{
                  position: 'absolute',
                  top: 0,
                  left: 0,
                  width: '100%',
                  height: '100%',
                  objectFit: 'cover',
                  borderRadius: '8px',
                  zIndex: 1,
                }}
                videoConstraints={{
                  width: 640,
                  height: 480,
                  facingMode: 'user'
                }}
              />
              <canvas
                ref={canvasRef}
                width={640}
                height={480}
                style={{
                  position: 'absolute',
                  top: 0,
                  left: 0,
                  width: '100%',
                  height: '100%',
                  borderRadius: '8px',
                  pointerEvents: 'none',
                  zIndex: 2,
                }}
              />
            </Box>
            {/* Controls */}
            <Box sx={{ mt: 2, display: 'flex', gap: 2, justifyContent: 'center', flexWrap: 'wrap', alignItems: 'center' }}>
              {!isActive ? (
                <>
                  <Button
                    variant="contained"
                    size="large"
                    startIcon={<PlayArrowIcon />}
                    onClick={handleStart}
                  >
                    Start Exercise
                  </Button>
                  <Button
                    variant="outlined"
                    size="medium"
                    onClick={() => setVoiceEnabled(!voiceEnabled)}
                    color={voiceEnabled ? 'success' : 'primary'}
                  >
                    🔊 Voice: {voiceEnabled ? 'ON' : 'OFF'}
                  </Button>
                </>
              ) : (
                <>
                  <Button
                    variant="contained"
                    size="large"
                    startIcon={isPaused ? <PlayArrowIcon /> : <PauseIcon />}
                    onClick={handlePause}
                  >
                    {isPaused ? 'Resume' : 'Pause'}
                  </Button>
                  <Button
                    variant="outlined"
                    size="large"
                    startIcon={<StopIcon />}
                    onClick={handleStop}
                  >
                    Stop
                  </Button>
                  <Button
                    variant="outlined"
                    size="medium"
                    onClick={() => setVoiceEnabled(!voiceEnabled)}
                    color={voiceEnabled ? 'success' : 'primary'}
                  >
                    🔊 {voiceEnabled ? 'ON' : 'OFF'}
                  </Button>
                </>
              )}
            </Box>
          </Paper>
        </Grid>

        {/* Stats and Info Cards */}
        <Grid item xs={12} md={5}>
          <Grid container spacing={1}>
            {/* Row 1: Main Stats */}
            <Grid item xs={12} sm={4}>
              <Card sx={{ 
                p: 1, 
                backgroundColor: repInProgress ? '#e3f2fd' : 'inherit',
                border: repInProgress ? '2px solid #2196f3' : 'inherit',
                transition: 'all 0.3s ease'
              }}>
                <CardContent sx={{ textAlign: 'center', p: 1 }}>
                  <Typography 
                    variant="h4" 
                    color="primary" 
                    gutterBottom
                    sx={{
                      fontWeight: 'bold',
                      animation: repInProgress ? 'pulse 1s infinite' : 'none',
                      '@keyframes pulse': {
                        '0%': { transform: 'scale(1)' },
                        '50%': { transform: 'scale(1.1)' },
                        '100%': { transform: 'scale(1)' }
                      }
                    }}
                  >
                    {repCount}
                  </Typography>
                  <Typography variant="body1" color="text.secondary">
                    Repetitions {repInProgress ? '(In Progress)' : ''}
                  </Typography>
                </CardContent>
              </Card>
            </Grid>
            <Grid item xs={12} sm={4}>
              <Card sx={{ p: 1 }}>
                <CardContent sx={{ textAlign: 'center', p: 1 }}>
                  <Typography variant="h5" color="secondary" gutterBottom>
                    {formatTime(sessionDuration)}
                  </Typography>
                  <Typography variant="body1" color="text.secondary">
                    Duration
                  </Typography>
                </CardContent>
              </Card>
            </Grid>
            <Grid item xs={12} sm={4}>
              <Card sx={{ p: 1 }}>
                <CardContent sx={{ textAlign: 'center', p: 1 }}>
                  <Typography variant="h6" gutterBottom>
                    AI Confidence
                  </Typography>
                  <LinearProgress 
                    variant="determinate" 
                    value={confidence * 100}
                    sx={{ mb: 1, height: 8, borderRadius: 4 }}
                    color={confidence > 0.8 ? 'success' : confidence > 0.6 ? 'warning' : 'error'}
                  />
                  <Typography variant="body2" color="text.secondary">
                    {(confidence * 100).toFixed(1)}% accuracy
                  </Typography>
                </CardContent>
              </Card>
            </Grid>

            {/* Row 2: Exercise Progress & Form Analysis */}
            <Grid item xs={12} sm={6}>
              <Card sx={{ p: 1 }}>
                <CardContent>
                  <Typography variant="h6" gutterBottom>
                    🎯 Exercise Progress
                  </Typography>
                  <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 1 }}>
                    <Typography variant="body1" fontWeight="medium">
                      Current Phase:
                    </Typography>
                    <Chip 
                      label={currentPhase.toUpperCase() || 'READY'} 
                      color={currentPhase ? 'success' : 'default'}
                      size="small"
                    />
                  </Box>
                  <Typography variant="body2" color="text.secondary">
                    Target Exercise: {selectedExercise.replace(/_/g, ' ').toUpperCase()}
                  </Typography>
                  <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mt: 1 }}>
                    <Typography variant="body2" fontWeight="medium">
                      Rep Status:
                    </Typography>
                    <Chip 
                      label={repInProgress ? 'IN PROGRESS' : 'READY'} 
                      color={repInProgress ? 'warning' : 'default'}
                      size="small"
                    />
                  </Box>
                </CardContent>
              </Card>
            </Grid>
            <Grid item xs={12} sm={6}>
              <Card sx={{ p: 1 }}>
                <CardContent>
                  <Typography variant="h6" gutterBottom>
                    📊 Form Analysis
                  </Typography>
                  <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 2 }}>
                    <span style={{ fontSize: '1.5em' }}>
                      {getFormQualityIcon(formQuality)}
                    </span>
                    <Chip 
                      label={formQuality.replace(/_/g, ' ').toUpperCase()} 
                      color={getFormQualityColor(formQuality) as any}
                      size="small"
                    />
                  </Box>
                  <Typography variant="body2" color="text.secondary" sx={{ 
                    backgroundColor: 'rgba(0,0,0,0.05)', 
                    p: 1, 
                    borderRadius: 1,
                    fontStyle: 'italic'
                  }}>
                    {exerciseFeedback || 'Start exercising to get feedback...'}
                  </Typography>
                </CardContent>
              </Card>
            </Grid>

            {/* Row 3: Tabs for Model Output, Log, Status */}
            <Grid item xs={12}>
              <Card sx={{ p: 1 }}>
                <CardContent sx={{ p: 1 }}>
                  <Tabs
                    value={tabValue}
                    onChange={(_, newValue) => setTabValue(newValue)}
                    variant="fullWidth"
                    sx={{ mb: 1 }}
                  >
                    <Tab label="AI Model Output" />
                    <Tab label="AI Processing Log" />
                    <Tab label="System Status" />
                  </Tabs>
                  {tabValue === 0 && (
                    <Box sx={{ 
                      backgroundColor: '#f5f5f5', 
                      p: 2, 
                      borderRadius: 1, 
                      fontFamily: 'monospace',
                      fontSize: '0.85em',
                      maxHeight: '200px',
                      overflowY: 'auto'
                    }}>
                      <Typography variant="body2" component="div">
                        <strong>🧠 Frame Buffer Status:</strong><br/>
                        <strong>Buffer Length:</strong> {frameBufferLength}/{SEQUENCE_LENGTH} frames<br/>
                        <strong>Keypoints Detected:</strong> {keypointsDetected} (expected: 33)<br/>
                        <strong>Vector Length:</strong> {keypointsDetected * 3} (expected: 99)<br/>
                        <strong>Buffer Ready:</strong> {frameBufferLength >= SEQUENCE_LENGTH ? '✅ Ready for prediction' : `⏳ Need ${SEQUENCE_LENGTH - frameBufferLength} more frames`}<br/>
                        <strong>Prediction Stability:</strong> {predictionHistory.length >= 3 ? '✅ Stabilized' : `⏳ Stabilizing (${predictionHistory.length}/3)`}<br/>
                        <strong>Rep Detection:</strong> {repInProgress ? '🏃‍♂️ In progress' : '⏸️ Waiting'}<br/>
                        <br/>
                        {aiModelDetails && (
                          <>
                            <strong>🤖 AI Debug Output (Hidden from User):</strong><br/>
                            <strong>Detected Exercise:</strong> {aiModelDetails.exercise}<br/>
                            <strong>Confidence:</strong> {(aiModelDetails.confidence * 100).toFixed(2)}%<br/>
                            <strong>Phase:</strong> {aiModelDetails.phase}<br/>
                            <strong>Backend Rep Count:</strong> {aiModelDetails.rep_count}<br/>
                            <strong>Status:</strong> {aiModelDetails.success ? '✅ Working' : '❌ Error'}<br/>
                            {aiModelDetails.timestamp && (
                              <>
                                <strong>Last Update:</strong> {new Date(aiModelDetails.timestamp).toLocaleTimeString()}<br/>
                              </>
                            )}
                          </>
                        )}
                      </Typography>
                    </Box>
                  )}
                  {tabValue === 1 && (
                    <Box sx={{ 
                      backgroundColor: '#1e1e1e', 
                      color: '#00ff00',
                      p: 2, 
                      borderRadius: 1, 
                      fontFamily: 'monospace',
                      fontSize: '0.8em',
                      height: '200px',
                      overflowY: 'auto',
                      border: '1px solid #333'
                    }}>
                      {consoleLog.length === 0 ? (
                        <Typography variant="body2" color="text.disabled">
                          Start exercising to see AI processing logs...
                        </Typography>
                      ) : (
                        consoleLog.map((entry, index) => (
                          <div key={index} style={{ marginBottom: '4px', lineHeight: '1.4' }}>
                            {entry}
                          </div>
                        ))
                      )}
                    </Box>
                  )}
                  {tabValue === 2 && (
                    <Box>
                      <Box sx={{ display: 'flex', flexDirection: 'column', gap: 1 }}>
                        <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                          <Typography variant="body2">Backend:</Typography>
                          <Chip 
                            label={error && error.includes('Backend') ? 'Error' : 'Connected'} 
                            color={error && error.includes('Backend') ? 'error' : 'success'}
                            size="small"
                          />
                        </Box>
                        <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                          <Typography variant="body2">Camera:</Typography>
                          <Chip 
                            label={isActive ? 'Active' : 'Inactive'} 
                            color={isActive ? 'success' : 'default'}
                            size="small"
                          />
                        </Box>
                        <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                          <Typography variant="body2">Pose Detection:</Typography>
                          <Chip 
                            label={poseDetected ? 'Detected' : 'Searching'} 
                            color={poseDetected ? 'success' : 'warning'}
                            size="small"
                          />
                        </Box>
                        <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                          <Typography variant="body2">Frame Buffer:</Typography>
                          <Chip 
                            label={`${frameBufferLength}/${SEQUENCE_LENGTH}`} 
                            color={frameBufferLength >= SEQUENCE_LENGTH ? 'success' : 'warning'}
                            size="small"
                          />
                        </Box>
                        <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                          <Typography variant="body2">Keypoints:</Typography>
                          <Chip 
                            label={`${keypointsDetected}/33`} 
                            color={keypointsDetected === 33 ? 'success' : 'warning'}
                            size="small"
                          />
                        </Box>
                        <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                          <Typography variant="body2">AI Model:</Typography>
                          <Chip 
                            label={confidence > 0.5 ? 'Active' : 'Standby'} 
                            color={confidence > 0.5 ? 'success' : 'default'}
                            size="small"
                          />
                        </Box>
                      </Box>
                      <Box sx={{ mt: 2 }}>
                        <Button 
                          variant="outlined" 
                          size="small" 
                          onClick={testConnection}
                          fullWidth
                          disabled={isActive}
                        >
                          🧪 Test Connection
                        </Button>
                      </Box>
                    </Box>
                  )}
                </CardContent>
              </Card>
            </Grid>
          </Grid>
        </Grid>
      </Grid>
    </Container>
  );
};

export default ExerciseMonitor; 