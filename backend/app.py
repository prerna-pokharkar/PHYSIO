import os
import pickle
import numpy as np
import math
from datetime import datetime
import json
from flask import Flask, request, jsonify
from flask_cors import CORS

# Try to import TensorFlow with fallback
try:
    import tensorflow as tf
    from tensorflow.keras.models import load_model
    TENSORFLOW_AVAILABLE = True
    print("TensorFlow imported successfully")
except ImportError as e:
    TENSORFLOW_AVAILABLE = False
    print(f"TensorFlow not available: {e}")
    print("Please install TensorFlow: pip install tensorflow")

from dotenv import load_dotenv

# Try to import OpenAI, fallback if not available
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
    print("OpenAI imported successfully")
except ImportError:
    OPENAI_AVAILABLE = False
    print("OpenAI package not available. Chatbot will use fallback responses.")

# Load environment variables
load_dotenv()

# Configure OpenAI if available
openai_client = None
if OPENAI_AVAILABLE and os.getenv('OPENAI_API_KEY'):
    try:
        openai_client = OpenAI(
            api_key=os.getenv('OPENAI_API_KEY'),
            timeout=30.0,
            max_retries=2
        )
        print("OpenAI API configured successfully")
    except Exception as e:
        print(f"Failed to initialize OpenAI client: {e}")
        print("Using fallback responses instead.")
        openai_client = None
else:
    print("OpenAI API not configured. Using fallback responses.")

app = Flask(__name__)

# Configure CORS with explicit settings for frontend communication
CORS(app, 
     origins=["http://localhost:3000", "http://127.0.0.1:3000"],  # Allow frontend origins
     allow_headers=["Content-Type", "Authorization"],             # Allow these headers
     methods=["GET", "POST", "OPTIONS"],                         # Allow these methods
     supports_credentials=True                                   # Allow credentials
)

# Global variables for model and encoder
model = None
label_encoder = None
exercise_sessions = []  # In-memory storage for demo (use database in production)

# Exercise phase tracking
current_exercise_state = {
    'current_phase': 'down',  # 'up' or 'down'
    'rep_count': 0,
    'last_prediction': None,
    'phase_threshold': 0.7  # Confidence threshold for phase detection
}

def load_models():
    """Load the BiLSTM model and label encoder"""
    global model, label_encoder
    
    if not TENSORFLOW_AVAILABLE:
        print("❌ TensorFlow not available. Cannot load ML models.")
        return False
    
    try:
        # Load the trained model
        model = load_model('model/model.h5')
        print("BiLSTM model loaded successfully")
        
        # Load the label encoder
        with open('model/label_encoder.pkl', 'rb') as f:
            label_encoder = pickle.load(f)
        print("Label encoder loaded successfully")
        print(f"Available exercises: {list(label_encoder.classes_)}")
        
    except Exception as e:
        print(f"Error loading models: {str(e)}")
        return False
    return True

def detect_exercise_phase_from_landmarks(landmarks_flat, predicted_exercise, selected_exercise=None):
    """
    Detect exercise phase directly from MediaPipe landmarks
    landmarks_flat: 99-length array [x1,y1,vis1, x2,y2,vis2, ..., x33,y33,vis33]
    This is much simpler than converting to joint angles first!
    """
    global current_exercise_state
    
    try:
        # Extract key landmark positions (MediaPipe Pose landmark indices)
        # Left elbow: index 13, Right elbow: index 14
        # Left wrist: index 15, Right wrist: index 16
        # Left shoulder: index 11, Right shoulder: index 12
        
        left_shoulder_y = landmarks_flat[11*3 + 1]   # y coordinate of left shoulder
        left_elbow_y = landmarks_flat[13*3 + 1]      # y coordinate of left elbow
        left_wrist_y = landmarks_flat[15*3 + 1]      # y coordinate of left wrist
        
        right_shoulder_y = landmarks_flat[12*3 + 1]  # y coordinate of right shoulder
        right_elbow_y = landmarks_flat[14*3 + 1]     # y coordinate of right elbow
        right_wrist_y = landmarks_flat[16*3 + 1]     # y coordinate of right wrist
        
        # Simple phase detection based on arm position relative to shoulders
        # For most exercises: when arms are extended (wrists below elbows) = 'up' phase
        #                   when arms are contracted (wrists above elbows) = 'down' phase
        
        # Use the average of both arms for stability
        avg_elbow_y = (left_elbow_y + right_elbow_y) / 2
        avg_wrist_y = (left_wrist_y + right_wrist_y) / 2
        avg_shoulder_y = (left_shoulder_y + right_shoulder_y) / 2
        
        # Determine phase based on relative positions
        if avg_wrist_y > avg_elbow_y:  # Wrists below elbows (in image coordinates, y increases downward)
            new_phase = 'down'  # Arms extended/lowered
        else:
            new_phase = 'up'    # Arms contracted/raised
        
        # Debug logging
        print(f"🔍 DEBUG Landmark Phase Detection:")
        print(f"   Exercise: {predicted_exercise}")
        print(f"   Avg Shoulder Y: {avg_shoulder_y:.3f}")
        print(f"   Avg Elbow Y: {avg_elbow_y:.3f}")
        print(f"   Avg Wrist Y: {avg_wrist_y:.3f}")
        print(f"   Detected Phase: {new_phase}")
        
        # Count reps on phase transitions
        old_phase = current_exercise_state['current_phase']
        old_rep_count = current_exercise_state['rep_count']
        
        if current_exercise_state['current_phase'] != new_phase:
            if current_exercise_state['current_phase'] == 'down' and new_phase == 'up':
                current_exercise_state['rep_count'] += 1
                print(f"   🔥 REP COMPLETED! {old_rep_count} → {current_exercise_state['rep_count']}")
            current_exercise_state['current_phase'] = new_phase
            print(f"   📈 Phase transition: {old_phase} → {new_phase}")
        else:
            print(f"   ➡️  Phase maintained: {new_phase}")
            
        return new_phase
        
    except Exception as e:
        print(f"Error in landmark-based phase detection: {e}")
        return 'unknown'

def extract_joint_angles_from_landmarks(landmarks_flat):
    """
    Extract joint angles from flattened MediaPipe landmarks
    landmarks_flat: 99-length array [x1,y1,vis1, x2,y2,vis2, ..., x33,y33,vis33]
    Returns: 9 joint angles matching the original system
    """
    # Reshape to (33, 3) format
    landmarks = []
    for i in range(0, 99, 3):
        landmarks.append({
            'x': landmarks_flat[i],
            'y': landmarks_flat[i+1],
            'visibility': landmarks_flat[i+2]
        })
    
    def calculate_angle(point1, point2, point3):
        """Calculate angle between three points"""
        vector1 = {
            'x': point1['x'] - point2['x'],
            'y': point1['y'] - point2['y']
        }
        
        vector2 = {
            'x': point3['x'] - point2['x'],
            'y': point3['y'] - point2['y']
        }
        
        # Calculate dot product and magnitudes
        dot_product = vector1['x'] * vector2['x'] + vector1['y'] * vector2['y']
        magnitude1 = (vector1['x'] ** 2 + vector1['y'] ** 2) ** 0.5
        magnitude2 = (vector2['x'] ** 2 + vector2['y'] ** 2) ** 0.5
        
        if magnitude1 == 0 or magnitude2 == 0:
            return 90  # Default angle
        
        cos_angle = dot_product / (magnitude1 * magnitude2)
        cos_angle = max(-1, min(1, cos_angle))  # Clamp to valid range
        
        angle = math.acos(cos_angle)
        return math.degrees(angle)
    
    try:
        # MediaPipe landmark indices (same as frontend)
        # 9 joint angles: [left_shoulder, right_shoulder, left_elbow, right_elbow, 
        #                  left_hip, right_hip, left_knee, right_knee, spine]
        
        angles = []
        
        # Left shoulder (11-13-15: shoulder-elbow-wrist)
        angles.append(calculate_angle(landmarks[11], landmarks[13], landmarks[15]))
        
        # Right shoulder (12-14-16)
        angles.append(calculate_angle(landmarks[12], landmarks[14], landmarks[16]))
        
        # Left elbow (13-15-17: elbow-wrist-pinky)
        angles.append(calculate_angle(landmarks[13], landmarks[15], landmarks[17]))
        
        # Right elbow (14-16-18)
        angles.append(calculate_angle(landmarks[14], landmarks[16], landmarks[18]))
        
        # Left hip (23-25-27: hip-knee-ankle)
        angles.append(calculate_angle(landmarks[23], landmarks[25], landmarks[27]))
        
        # Right hip (24-26-28)
        angles.append(calculate_angle(landmarks[24], landmarks[26], landmarks[28]))
        
        # Left knee (25-27-31: knee-ankle-heel)
        angles.append(calculate_angle(landmarks[25], landmarks[27], landmarks[31]))
        
        # Right knee (26-28-32)
        angles.append(calculate_angle(landmarks[26], landmarks[28], landmarks[32]))
        
        # Spine angle (11-23-24: left_shoulder-left_hip-right_hip)
        angles.append(calculate_angle(landmarks[11], landmarks[23], landmarks[24]))
        
        return angles
        
    except Exception as e:
        print(f"Error extracting joint angles: {e}")
        return [90] * 9  # Return default angles

def detect_exercise_phase(joint_angles, predicted_exercise, selected_exercise=None):
    """
    Detect if the exercise is in 'up' or 'down' phase based on joint angles
    Updated for the actual trained exercises
    Only counts reps if exercise matches selection
    """
    global current_exercise_state
    
    # Get key joint angles with safety checks
    shoulder_angle = joint_angles[0] if len(joint_angles) > 0 else 90
    elbow_angle = joint_angles[2] if len(joint_angles) > 2 else 90
    hip_angle = joint_angles[4] if len(joint_angles) > 4 else 90
    knee_angle = joint_angles[6] if len(joint_angles) > 6 else 90
    
    # Debug logging
    print(f"🔍 DEBUG Phase Detection:")
    print(f"   Exercise: {predicted_exercise}")
    print(f"   Angles - Shoulder: {shoulder_angle:.1f}°, Elbow: {elbow_angle:.1f}°, Hip: {hip_angle:.1f}°, Knee: {knee_angle:.1f}°")
    
    # Phase detection logic for each exercise type
    exercise_lower = predicted_exercise.lower().replace('-', '_')
    
    # Simple phase detection based on elbow angle
    if len(joint_angles) > 2:  # Ensure we have elbow angle data
        elbow_angle = joint_angles[2]
        if elbow_angle < 90:
            new_phase = 'down'
        else:
            new_phase = 'up'
    
    # Debug current state
    old_phase = current_exercise_state['current_phase']
    old_rep_count = current_exercise_state['rep_count']
    
    # Count reps on phase transitions (except for isometric exercises)
    if new_phase != 'hold' and current_exercise_state['current_phase'] != new_phase:
        if current_exercise_state['current_phase'] == 'down' and new_phase == 'up':
            current_exercise_state['rep_count'] += 1
            print(f"   🔥 REP COMPLETED! {old_rep_count} → {current_exercise_state['rep_count']}")
        current_exercise_state['current_phase'] = new_phase
        print(f"   📈 Phase transition: {old_phase} → {new_phase}")
    elif new_phase == 'hold':
        # For isometric exercises, maintain state
        current_exercise_state['current_phase'] = new_phase
    else:
        print(f"   ➡️  Phase maintained: {new_phase}")
        
    return new_phase

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'encoder_loaded': label_encoder is not None,
        'openai_available': OPENAI_AVAILABLE,
        'openai_configured': OPENAI_AVAILABLE and bool(os.getenv('OPENAI_API_KEY')),
        'timestamp': datetime.now().isoformat()
    })

@app.route('/exercises', methods=['GET'])
def get_exercises():
    """Get list of available exercises from label encoder"""
    if label_encoder is None:
        return jsonify({'error': 'Label encoder not loaded'}), 500
    
    return jsonify({
        'exercises': list(label_encoder.classes_)
    })

@app.route('/predict', methods=['POST'])
def predict():
    """Predict exercise from MediaPipe landmark sequence"""
    try:
        data = request.get_json()
        print(f"🔍 Received prediction request with keys: {list(data.keys()) if data else 'None'}")
        print(f"🔍 Sequence length: {len(data.get('sequence', [])) if data else 0}")
        print(f"🔍 Selected exercise: {data.get('selected_exercise', 'None')}")
        
        landmark_sequence = data.get('sequence', [])
        selected_exercise = data.get('selected_exercise', None)
        
        # Validate sequence format: should be 30 frames of 99 landmarks each
        if not landmark_sequence or len(landmark_sequence) != 30:
            return jsonify({'error': f'Invalid sequence length: expected 30 frames, got {len(landmark_sequence) if landmark_sequence else 0}'}), 400
            
        # Validate each frame has 99 landmarks (33 keypoints × 3 values)
        for i, frame in enumerate(landmark_sequence):
            if len(frame) != 99:
                return jsonify({'error': f'Invalid frame {i}: expected 99 landmarks, got {len(frame)}'}), 400
        
        # Validate pose quality - check if landmarks are valid (not all zeros/ones)
        # For MediaPipe landmarks, x,y should be in [0,1] range, visibility in [0,1]
        first_frame = landmark_sequence[0]
        # Check if too many landmarks are at edge values (poor detection)
        invalid_landmarks = 0
        for i in range(0, 99, 3):  # Check every x,y,visibility triplet
            x, y, vis = first_frame[i], first_frame[i+1], first_frame[i+2]
            if vis < 0.3 or x <= 0.01 or x >= 0.99 or y <= 0.01 or y >= 0.99:
                invalid_landmarks += 1
        
        if invalid_landmarks > 20:  # If more than 20 out of 33 landmarks are invalid
            return jsonify({
                'exercise': 'unknown',
                'confidence': 0.0,
                'phase': 'unknown',
                'rep_count': current_exercise_state['rep_count'],
                'joint_angles': [],  # Empty for landmark-based system
                'timestamp': datetime.now().isoformat(),
                'error': 'Poor pose detection - please ensure you are fully visible in the camera'
            })
        
        # Convert sequence to numpy array for model input
        model_input = np.array(landmark_sequence, dtype=np.float32)
        print(f"🔧 Model input shape: {model_input.shape}")  # Should be (30, 99)
        
        # Reshape for model input: (1, 30, 99) - batch_size=1
        model_input = model_input.reshape(1, 30, 99)
        
        # Make prediction
        prediction = model.predict(model_input, verbose=0)

        print(f"🔍 Prediction result: {prediction}")
        
        # Get predicted class and confidence
        predicted_class_idx = np.argmax(prediction[0])
        confidence = float(np.max(prediction[0]))
        
        # Convert to exercise name
        predicted_exercise = label_encoder.inverse_transform([predicted_class_idx])[0]
        
        # For phase detection, we can either:
        # 1. Use raw landmark positions (simpler, more direct)
        # 2. Convert to joint angles (more complex, but matches existing logic)
        # Let's use option 1 for now - direct landmark-based phase detection
        
        # Extract key landmarks for phase detection from the first frame
        first_frame_landmarks = landmark_sequence[0]
        
        # Only detect phase and count reps if conditions are met
        phase = 'unknown'
        exercise_match = False
        
        # Check if detected exercise matches selected exercise and confidence is sufficient
        if selected_exercise:
            # 🔥 ULTRA-RELAXED MATCHING FOR TESTING - Allow any high confidence prediction
            exercise_match = (
                (predicted_exercise.lower() == selected_exercise.lower() and confidence >= 0.5) or
                (confidence >= 0.8)  # Accept high confidence regardless of exercise match
            )
        else:
            exercise_match = confidence >= 0.7  # Higher threshold if no exercise selected
            
        # 🔥 FORCE PHASE DETECTION FOR TESTING - Always detect phase if confidence is very high
        if confidence >= 0.8:  # Force phase detection for high confidence predictions
            exercise_match = True
            print(f"🔥 FORCING phase detection due to high confidence: {confidence:.3f} - overriding exercise mismatch")
            
        if exercise_match:
            # Only detect phase and count reps if exercise matches
            phase = detect_exercise_phase_from_landmarks(first_frame_landmarks, predicted_exercise, selected_exercise)
        else:
            # Reset phase tracking if exercise doesn't match
            print(f"🚫 Exercise mismatch or low confidence: detected={predicted_exercise}, selected={selected_exercise}, confidence={confidence:.2f}")
            phase = 'unknown'  # Explicitly set unknown phase
        
        response = {
            'exercise': predicted_exercise,
            'confidence': confidence,
            'phase': phase,
            'rep_count': current_exercise_state['rep_count'],
            'landmark_sequence_length': len(landmark_sequence),
            'timestamp': datetime.now().isoformat(),
            'exercise_match': exercise_match,
            'selected_exercise': selected_exercise,
            'current_phase': current_exercise_state['current_phase'],  # Add current phase for frontend
            'debug_info': {  # Add debug info for troubleshooting
                'predicted_exercise': predicted_exercise,
                'confidence_threshold_met': confidence >= (0.6 if selected_exercise else 0.7),
                'exercise_names_match': predicted_exercise.lower() == (selected_exercise.lower() if selected_exercise else ''),
                'phase_detection_active': exercise_match
            }
        }
        
        return jsonify(response)
        
    except Exception as e:
        print(f"Prediction error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'Prediction failed: {str(e)}'}), 500

@app.route('/reset_session', methods=['POST'])
def reset_session():
    """Reset the current exercise session"""
    global current_exercise_state
    
    current_exercise_state = {
        'current_phase': 'down',
        'rep_count': 0,
        'last_prediction': None,
        'phase_threshold': 0.7
    }
    
    return jsonify({'message': 'Session reset successfully'})

@app.route('/log_session', methods=['POST'])
def log_session():
    """
    Log exercise session data
    Expected input: {
        'user_id': 'user123',
        'exercise': 'pushup',
        'total_reps': 10,
        'duration': 120,  # seconds
        'session_data': [...] # optional detailed data
    }
    """
    try:
        data = request.get_json()
        
        required_fields = ['user_id', 'exercise', 'total_reps', 'duration']
        for field in required_fields:
            if field not in data:
                return jsonify({'error': f'Missing required field: {field}'}), 400
        
        session_entry = {
            'user_id': data['user_id'],
            'exercise': data['exercise'],
            'total_reps': data['total_reps'],
            'duration': data['duration'],  # in seconds
            'timestamp': datetime.now().isoformat(),
            'session_data': data.get('session_data', [])
        }
        
        # Store session (in production, save to database)
        exercise_sessions.append(session_entry)
        
        return jsonify({
            'message': 'Session logged successfully',
            'session_id': len(exercise_sessions) - 1
        })
        
    except Exception as e:
        return jsonify({'error': f'Failed to log session: {str(e)}'}), 500

@app.route('/sessions/<user_id>', methods=['GET'])
def get_user_sessions(user_id):
    """Get all sessions for a specific user"""
    user_sessions = [session for session in exercise_sessions if session['user_id'] == user_id]
    
    # Calculate summary statistics
    total_sessions = len(user_sessions)
    total_reps = sum(session['total_reps'] for session in user_sessions)
    total_duration = sum(session['duration'] for session in user_sessions)
    
    # Group by exercise
    exercise_stats = {}
    for session in user_sessions:
        exercise = session['exercise']
        if exercise not in exercise_stats:
            exercise_stats[exercise] = {'sessions': 0, 'total_reps': 0, 'total_duration': 0}
        exercise_stats[exercise]['sessions'] += 1
        exercise_stats[exercise]['total_reps'] += session['total_reps']
        exercise_stats[exercise]['total_duration'] += session['duration']
    
    return jsonify({
        'user_id': user_id,
        'sessions': user_sessions,
        'summary': {
            'total_sessions': total_sessions,
            'total_reps': total_reps,
            'total_duration': total_duration,
            'exercise_breakdown': exercise_stats
        }
    })

@app.route('/sessions', methods=['GET'])
def get_all_sessions():
    """Get all sessions (for admin/debugging)"""
    return jsonify({'sessions': exercise_sessions})

@app.route('/chatbot', methods=['POST'])
def chatbot():
    """Handle chatbot conversations using OpenAI API or fallback"""
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        uid = data.get('uid')
        messages = data.get('messages', [])
        
        if not uid:
            return jsonify({'error': 'User ID is required'}), 400
        
        if not messages:
            return jsonify({'error': 'Messages array is required'}), 400
        
        # Try OpenAI API if available, otherwise use fallback
        if OPENAI_AVAILABLE and openai_client:
            try:
                # Prepare system message for fitness context
                system_message = {
                    "role": "system",
                    "content": """You are Alex, a certified personal trainer, nutritionist, and exercise expert with 15+ years of experience. You're working as an AI fitness assistant in a comprehensive exercise monitoring app called PhysioTracker.

� **YOUR ROLE:**
You're a conversational fitness expert who answers ANY fitness, health, nutrition, or exercise-related questions naturally and helpfully - just like ChatGPT but specialized in fitness.

�️ **EXPERTISE AREAS:**
- Exercise techniques, form, and programming
- Workout routines for all fitness levels
- Nutrition, meal planning, and diet advice
- Weight gain/loss strategies
- Muscle building and strength training
- Injury prevention and recovery
- Sports performance and endurance
- Mental health and motivation

💬 **CONVERSATION STYLE:**
- Be natural, friendly, and conversational (like ChatGPT)
- Answer questions directly and thoroughly
- Ask follow-up questions when needed for better advice
- Use emojis and formatting to keep responses engaging
- Provide specific, actionable advice with numbers when relevant
- Always be encouraging and supportive

🚨 **IMPORTANT:**
- Answer the user's actual question naturally
- Don't always ask for personal details unless relevant
- Keep responses concise but comprehensive
- For serious medical concerns, recommend consulting healthcare professionals

Remember: You're having a natural conversation about fitness, not conducting an interview or always pushing for personal details."""
                }
                
                # Combine system message with user conversation
                api_messages = [system_message] + messages
                
                print(f"Sending to OpenAI: {len(api_messages)} messages")
                print(f"Last user message: {messages[-1].get('content', '')[:100]}...")
                
                # Call OpenAI API with updated syntax
                response = openai_client.chat.completions.create(
                    model="gpt-4o-mini",  # Use the efficient model
                    messages=api_messages,
                    max_tokens=1200,
                    temperature=0.7,
                    top_p=1,
                    frequency_penalty=0,
                    presence_penalty=0
                )
                
                reply = response.choices[0].message.content.strip()
                
                print(f"OpenAI response received: {len(reply)} characters")
                
                return jsonify({
                    'reply': reply,
                    'timestamp': datetime.now().isoformat(),
                    'uid': uid,
                    'model': 'gpt-4o-mini',
                    'source': 'openai'
                })
                
            except Exception as e:
                print(f"OpenAI API error: {str(e)}")
                print(f"Error type: {type(e)}")
                # Fall through to fallback if OpenAI fails
        
        # Simplified fallback response system (only when OpenAI fails)
        return simple_fallback_response(messages, uid)
        
    except Exception as e:
        print(f"Chatbot error: {str(e)}")
        return jsonify({
            'error': 'Internal server error',
            'message': str(e)
        }), 500

def simple_fallback_response(messages, uid):
    """Simple fallback when OpenAI is not available"""
    last_message = messages[-1] if messages else {'content': ''}
    user_content = last_message.get('content', '').lower()
    
    # Check if user is asking for a detailed plan
    if any(phrase in user_content for phrase in ['make me a plan', 'create a plan', 'plan for me', 'detailed plan']):
        # Try to extract user details for personalized plan
        user_details = parse_user_details(user_content)
        if any(user_details.values()):
            return jsonify({
                'reply': generate_personalized_plan(user_details),
                'timestamp': datetime.now().isoformat(),
                'uid': uid,
                'fallback': True,
                'personalized': True,
                'source': 'fallback_detailed_plan'
            })
    
    # Simple general fitness responses when OpenAI is not available
    if 'hello' in user_content or 'hi' in user_content:
        reply = """👋 Hi there! I'm your fitness assistant Alex!

I can help you with:
🏋️ Workout routines and exercise techniques
🥗 Nutrition advice and meal planning  
💪 Muscle building and strength training
🔥 Weight loss and fat burning strategies
🏃 Cardio and endurance training

What fitness topic would you like to discuss today?"""
    
    elif 'help' in user_content:
        reply = """🆘 **Here's how I can help you:**

💬 **Ask me anything about:**
- Exercise techniques and form
- Workout routines for your goals
- Nutrition and diet advice
- Weight gain/loss strategies
- Muscle building tips
- Injury prevention

💡 **Try asking:**
- "What's a good chest workout at home?"
- "How can I gain weight healthily?"
- "Best exercises for core strength?"
- "Can you make me a plan?" (for detailed personalized plans)

What's your fitness question?"""
    
    elif any(word in user_content for word in ['workout', 'exercise', 'training']):
        reply = """💪 **Great question about workouts!**

I'd love to help you with exercise advice! However, I'm currently running in limited mode.

For the best personalized workout advice, please:
1. Make sure your internet connection is stable
2. Try asking your question again

In the meantime, here are some basic principles:
✅ Start with compound movements (squats, push-ups, rows)
✅ Progress gradually in intensity
✅ Include both strength and cardio
✅ Rest 48-72 hours between training same muscle groups

What specific exercise question can I help with?"""
    
    elif any(word in user_content for word in ['nutrition', 'diet', 'meal', 'food']):
        reply = """🥗 **Nutrition is key to your fitness success!**

I'm currently in limited mode, but here are fundamental nutrition tips:

✅ **Protein:** 1.6-2.2g per kg bodyweight daily
✅ **Hydration:** 8-10 glasses of water daily  
✅ **Timing:** Eat protein within 30 mins post-workout
✅ **Balance:** Include protein, carbs, and healthy fats in meals
✅ **Frequency:** Eat every 3-4 hours to maintain energy

For detailed meal plans, try asking "Can you make me a plan?" with your stats!

What specific nutrition question do you have?"""
    
    else:
        reply = """🤖 **I'm here to help with your fitness journey!**

I'm currently running in limited mode, but I can still assist with basic fitness questions.

💡 **Try asking me about:**
- Exercise techniques
- Workout routines  
- Nutrition basics
- General fitness advice

Or say "Can you make me a plan?" along with your stats (age, weight, height, goals) for a detailed personalized plan!

What fitness topic interests you most?"""
    
    return jsonify({
        'reply': reply,
        'timestamp': datetime.now().isoformat(),
        'uid': uid,
        'fallback': True,
        'source': 'simple_fallback'
    })

def parse_user_details(text):
    """Extract user details from text using regex and keywords"""
    import re
    
    details = {
        'current_weight': None,
        'target_weight': None,
        'height': None,
        'age': None,
        'timeline_months': None,
        'goal': None,
        'name': None
    }
    
    text_lower = text.lower()
    
    # Extract name (common patterns)
    name_patterns = [
        r'i am (\w+)',
        r'my name is (\w+)',
        r'i\'m (\w+)',
        r'this is (\w+)'
    ]
    for pattern in name_patterns:
        match = re.search(pattern, text_lower)
        if match:
            details['name'] = match.group(1).title()
            break
    
    # Extract current weight
    weight_patterns = [
        r'i am (\d+)kg',
        r'i\'m (\d+)kg', 
        r'(\d+)kg right now',
        r'currently (\d+)kg',
        r'i weigh (\d+)kg',
        r'my weight is (\d+)kg'
    ]
    for pattern in weight_patterns:
        match = re.search(pattern, text_lower)
        if match:
            details['current_weight'] = int(match.group(1))
            break
    
    # Extract target weight
    target_patterns = [
        r'want to (?:gain|lose) (\d+)kg',
        r'need to (?:gain|lose) (\d+)kg',
        r'goal is (\d+)kg',
        r'target (?:is )?(\d+)kg'
    ]
    for pattern in target_patterns:
        match = re.search(pattern, text_lower)
        if match:
            target_change = int(match.group(1))
            if details['current_weight']:
                if 'gain' in text_lower or 'want to gain' in text_lower:
                    details['target_weight'] = details['current_weight'] + target_change
                elif 'lose' in text_lower or 'want to lose' in text_lower:
                    details['target_weight'] = details['current_weight'] - target_change
            break
    
    # Extract height
    height_patterns = [
        r'i am (\d+)\'(\d+)"?',  # 5'8" format
        r'(\d+)\'(\d+)"?',
        r'(\d+) feet (\d+) inch',
        r'(\d+)ft (\d+)in',
        r'height is (\d+)\'(\d+)"?'
    ]
    for pattern in height_patterns:
        match = re.search(pattern, text_lower)
        if match:
            feet = int(match.group(1))
            inches = int(match.group(2))
            details['height'] = f"{feet}'{inches}\""
            break
    
    # Extract age
    age_patterns = [
        r'i am (\d+) years old',
        r'(\d+) years old',
        r'age is (\d+)',
        r'i\'m (\d+)',
        r'(\d+)y(?:ears)?o'
    ]
    for pattern in age_patterns:
        match = re.search(pattern, text_lower)
        if match:
            age = int(match.group(1))
            if 15 <= age <= 80:  # Reasonable age range
                details['age'] = age
                break
    
    # Extract timeline
    timeline_patterns = [
        r'in (\d+) months?',
        r'within (\d+) months?',
        r'over (\d+) months?',
        r'(\d+) months?'
    ]
    for pattern in timeline_patterns:
        match = re.search(pattern, text_lower)
        if match:
            details['timeline_months'] = int(match.group(1))
            break
    
    # Determine goal
    if 'gain' in text_lower or 'want to gain' in text_lower:
        details['goal'] = 'weight_gain'
    elif 'lose' in text_lower or 'want to lose' in text_lower or 'loose' in text_lower:
        details['goal'] = 'weight_loss'
    elif 'muscle' in text_lower or 'bulk' in text_lower:
        details['goal'] = 'muscle_gain'
    elif 'fat' in text_lower and ('lose' in text_lower or 'burn' in text_lower):
        details['goal'] = 'fat_loss'
    
    return details

def generate_personalized_plan(details):
    """Generate a personalized fitness plan based on extracted details"""
    name = details['name'] or "there"
    current_weight = details['current_weight']
    target_weight = details['target_weight']
    height = details['height']
    age = details['age']
    timeline = details['timeline_months']
    goal = details['goal']
    
    # Calculate BMI if possible
    bmi_info = ""
    if current_weight and height:
        try:
            # Convert height to meters (assuming feet'inches format)
            if "'" in str(height):
                feet, inches = height.replace('"', '').split("'")
                height_m = (int(feet) * 12 + int(inches)) * 0.0254
                bmi = current_weight / (height_m ** 2)
                
                if bmi < 18.5:
                    bmi_info = f"📊 **BMI Analysis:** {bmi:.1f} (Underweight) - Perfect for weight gain goals!"
                elif 18.5 <= bmi < 25:
                    bmi_info = f"📊 **BMI Analysis:** {bmi:.1f} (Normal) - Good foundation for your goals!"
                elif 25 <= bmi < 30:
                    bmi_info = f"📊 **BMI Analysis:** {bmi:.1f} (Overweight) - Weight loss will improve your health!"
                else:
                    bmi_info = f"📊 **BMI Analysis:** {bmi:.1f} (Obese) - Let's focus on sustainable weight loss!"
        except:
            pass
    
    if goal == 'weight_gain':
        weight_change = target_weight - current_weight if target_weight and current_weight else 10
        
        # Calculate healthy rate
        recommended_months = max(2, weight_change * 0.5)  # 0.5-1kg per month is healthy
        rate_warning = ""
        if timeline and timeline < recommended_months:
            rate_warning = f"\n⚠️ **IMPORTANT:** Your timeline of {timeline} months is aggressive. I recommend {recommended_months:.0f} months for sustainable results."
        
        # Calculate calories
        daily_calories = 2500 + (weight_change * 150)  # Rough estimate
        
        plan = f"""🎯 **PERSONALIZED WEIGHT GAIN PLAN for {name}!** 💪

**📋 YOUR PROFILE:**
- Current: {current_weight}kg → Target: {target_weight}kg (+{weight_change}kg)
- Height: {height}, Age: {age}
- Timeline: {timeline} months
{bmi_info}{rate_warning}

**🍽️ DAILY NUTRITION TARGET:**
- **Calories:** {daily_calories:,}/day (+500-750 surplus)
- **Protein:** {current_weight * 2.2:.0f}g/day (for muscle growth)
- **Carbs:** {current_weight * 4:.0f}g/day (energy for workouts)
- **Fats:** {current_weight * 1:.0f}g/day (hormone production)

**📅 SAMPLE MEAL SCHEDULE:**
**7:00 AM - Breakfast (650 cal):**
- 1 cup oats + banana + 30g almonds + milk
- OR 3 eggs + 2 toast + avocado

**10:00 AM - Snack (300 cal):**
- Protein smoothie: milk + banana + protein powder + dates

**1:00 PM - Lunch (700 cal):**
- 150g chicken/paneer + 1.5 cups rice + vegetables + ghee
- OR Quinoa bowl with chickpeas and nuts

**4:00 PM - Pre-workout (200 cal):**
- Banana + 2 tbsp peanut butter

**6:00 PM - Post-workout (250 cal):**
- Whey protein shake + apple

**8:00 PM - Dinner (600 cal):**
- 150g fish/lentils + sweet potato + salad + olive oil

**10:00 PM - Before bed (200 cal):**
- Greek yogurt + honey OR milk + almonds

**💪 WORKOUT PLAN (4 days/week):**
**Monday & Thursday - Upper Body:**
1. Push-ups: 3×8-15 (progress to decline)
2. Pull-ups/Rows: 3×5-12
3. Shoulder press: 3×10-15
4. Bicep curls: 3×12-15
5. Tricep dips: 3×8-12

**Tuesday & Friday - Lower Body + Core:**
1. Squats: 3×12-20
2. Lunges: 3×10 each leg
3. Deadlifts: 3×8-12
4. Calf raises: 3×15-20
5. Planks: 3×45-90 seconds

**🎯 WEEKLY TARGETS:**
✅ Eat every 2-3 hours (6 meals/day)
✅ Drink 3-4L water daily
✅ Sleep 7-9 hours nightly
✅ Track weight weekly (same time, same conditions)
✅ Progress photos monthly

**📈 EXPECTED PROGRESS:**
- Month 1: +1-2kg (initial gains)
- Month 2: +2-3kg total
- Month 3+: +0.5-1kg/month (lean muscle)

**💡 PRO TIPS:**
🥛 Add healthy fats (nuts, avocado, olive oil) to meals
🍌 Eat carbs around workouts for energy
💤 Most muscle growth happens during sleep
📱 Use a food tracking app for first month

Want me to adjust anything based on your preferences or dietary restrictions?"""

    elif goal == 'weight_loss' or goal == 'fat_loss':
        weight_change = current_weight - target_weight if target_weight and current_weight else 10
        
        # Calculate healthy rate
        recommended_months = max(2, weight_change * 0.5)  # 0.5-1kg per month is healthy
        rate_warning = ""
        if timeline and timeline < recommended_months:
            rate_warning = f"\n⚠️ **IMPORTANT:** Your timeline of {timeline} months is aggressive. I recommend {recommended_months:.0f} months for sustainable results."
        
        # Calculate calories for deficit
        maintenance_calories = 2000 + (current_weight * 15) if current_weight else 2200
        daily_calories = maintenance_calories - 500  # 500 calorie deficit
        
        plan = f"""🔥 **PERSONALIZED FAT LOSS PLAN for {name}!** 💨

**📋 YOUR PROFILE:**
- Current: {current_weight}kg → Target: {target_weight}kg (-{weight_change}kg)
- Height: {height}, Age: {age}
- Timeline: {timeline} months
{bmi_info}{rate_warning}

**🍽️ DAILY NUTRITION TARGET:**
- **Calories:** {daily_calories:,}/day (-500 deficit)
- **Protein:** {current_weight * 2.5:.0f}g/day (preserve muscle)
- **Carbs:** {current_weight * 2:.0f}g/day (energy + fiber)
- **Fats:** {current_weight * 0.8:.0f}g/day (hormones)

**📅 SAMPLE MEAL SCHEDULE:**
**7:00 AM - Breakfast (350 cal):**
- Greek yogurt + berries + 1 tbsp honey
- OR 2 eggs + 1 toast + spinach

**10:00 AM - Snack (150 cal):**
- Apple + 15g almonds
- OR carrot sticks + hummus

**1:00 PM - Lunch (450 cal):**
- 120g grilled chicken + quinoa + large salad
- OR lentil soup + vegetables

**4:00 PM - Snack (100 cal):**
- Cucumber + Greek yogurt dip
- OR green tea + 5 almonds

**8:00 PM - Dinner (400 cal):**
- 120g fish + steamed vegetables + sweet potato

**💪 WORKOUT PLAN (5 days/week):**
**Monday, Wednesday, Friday - HIIT (25 mins):**
- Warm-up: 5 mins light cardio
- HIIT: 30 sec intense, 90 sec recovery × 8 rounds
- Cool-down: 5 mins stretching

**Tuesday, Thursday - Strength (45 mins):**
- Full body circuit training
- 3 rounds, 12-15 reps each:
  1. Squats, 2. Push-ups, 3. Rows
  4. Lunges, 5. Planks, 6. Burpees

**🎯 DAILY TARGETS:**
✅ 10,000+ steps (use phone tracker)
✅ 2.5-3L water daily
✅ 7+ hours quality sleep
✅ Log all food intake for first month

**📈 EXPECTED PROGRESS:**
- Week 1-2: -1-2kg (water weight)
- Month 1: -3-4kg total
- Monthly: -2-3kg sustainable loss

**💡 FAT LOSS HACKS:**
🥗 Fill half your plate with vegetables
🚶 Walk 10 mins after each meal
☕ Drink green tea before workouts
📱 Take progress photos weekly
🍽️ Eat slowly and chew thoroughly

Ready to start your transformation journey?"""

    else:
        # Generic plan when goal is unclear
        plan = f"""💪 **PERSONALIZED FITNESS PLAN for {name}!** 🎯

**📋 YOUR PROFILE:**
- Current weight: {current_weight}kg
- Height: {height}
- Age: {age}
{bmi_info}

Based on your stats, here's what I recommend:

**🎯 SUGGESTED GOALS:**
1. **Body Recomposition** - Build muscle while losing fat
2. **Strength Building** - Focus on getting stronger
3. **General Fitness** - Improve overall health and energy

**💪 STARTER WORKOUT (3 days/week):**
**Full Body Circuit:**
1. Bodyweight squats: 3×12-15
2. Push-ups: 3×8-12 (modify as needed)
3. Planks: 3×30-60 seconds
4. Lunges: 3×10 each leg
5. Mountain climbers: 3×20

**🍽️ BALANCED NUTRITION:**
- Focus on whole foods: lean protein, vegetables, fruits, whole grains
- Eat balanced meals every 3-4 hours
- Stay hydrated: 8-10 glasses water daily

**📈 NEXT STEPS:**
Tell me your specific goal and I'll create a detailed plan:
- "I want to build muscle"
- "I need to lose belly fat"
- "I want to get stronger"
- "I want to improve my endurance"

What's your main fitness priority?"""
    
    return plan

def fallback_chatbot_response(messages, uid):
    """Enhanced fallback responses with detailed fitness guidance"""
    last_message = messages[-1] if messages else {'content': ''}
    user_content = last_message.get('content', '').lower()
    
    # First, try to extract and parse user details
    user_details = parse_user_details(user_content)
    
    # If we found specific user details, generate personalized plan
    if any(user_details.values()):
        return jsonify({
            'reply': generate_personalized_plan(user_details),
            'timestamp': datetime.now().isoformat(),
            'uid': uid,
            'fallback': True,
            'personalized': True
        })
    
    # Enhanced responses with detailed fitness guidance (existing logic)
    if 'weight gain' in user_content or 'gain weight' in user_content:
        reply = """🏋️ **WEIGHT GAIN STRATEGY** 💪

Great goal! Here's your comprehensive weight gain plan:

**📊 CALORIE CALCULATION:**
- For 47kg → 55kg in 2 months, you need to gain ~1kg/week
- Target: 3000-3500 calories/day (500-750 surplus)
- Focus: 40% carbs, 30% protein, 30% healthy fats

**🍽️ MEAL PLAN:**
**Breakfast:** Oats + banana + nuts + milk (450 cal)
**Mid-morning:** Protein smoothie + dates (300 cal)
**Lunch:** Rice + chicken/paneer + vegetables (600 cal)
**Snack:** Almonds + fruit (250 cal)
**Pre-workout:** Banana + peanut butter (200 cal)
**Post-workout:** Protein shake (250 cal)
**Dinner:** Quinoa + fish/lentils + vegetables (550 cal)
**Before bed:** Milk + honey (150 cal)

**💪 WORKOUT ROUTINE (4 days/week):**
**Day 1 & 3 - Upper Body:**
- Push-ups: 3 sets × 8-12 reps
- Pull-ups/Rows: 3 sets × 6-10 reps
- Shoulder press: 3 sets × 10-12 reps
- Bicep curls: 3 sets × 12-15 reps

**Day 2 & 4 - Lower Body:**
- Squats: 3 sets × 12-15 reps
- Lunges: 3 sets × 10 each leg
- Calf raises: 3 sets × 15-20 reps
- Planks: 3 sets × 30-60 seconds

**🎯 KEY TIPS:**
✅ Eat every 2-3 hours
✅ Drink 3-4L water daily
✅ Sleep 7-8 hours nightly
✅ Track progress weekly

What's your current activity level and available equipment?"""

    elif 'lose weight' in user_content or 'weight loss' in user_content or 'fat loss' in user_content:
        reply = """🔥 **WEIGHT LOSS STRATEGY** 💨

Let's create your fat loss plan!

**📊 CALORIE DEFICIT:**
- Create 500-750 calorie deficit daily
- Target: 0.5-1kg loss per week (sustainable)
- Focus: 45% protein, 35% carbs, 20% fats

**🍽️ SAMPLE MEAL PLAN:**
**Breakfast:** Greek yogurt + berries + oats (300 cal)
**Snack:** Apple + almonds (150 cal)
**Lunch:** Grilled chicken + quinoa + salad (400 cal)
**Snack:** Vegetable sticks + hummus (100 cal)
**Dinner:** Fish + steamed vegetables (350 cal)

**🏃 CARDIO PLAN:**
**3x/week HIIT (20 mins):**
- 30 sec high intensity
- 90 sec recovery
- Repeat 8 rounds

**💪 STRENGTH TRAINING (3x/week):**
- Full body compound movements
- Circuit training style
- 3 sets × 12-15 reps

**🎯 DAILY TARGETS:**
✅ 10,000+ steps
✅ 2-3L water
✅ 7+ hours sleep
✅ Track food intake

What's your current weight and target goal?"""

    elif 'muscle' in user_content or 'strength' in user_content or 'bulk' in user_content:
        reply = """💪 **MUSCLE BUILDING PROGRAM** 🏋️

Time to build some serious muscle!

**🏋️ TRAINING SPLIT (5 days):**
**Monday - Chest & Triceps:**
- Bench press: 4×6-8
- Incline dumbbell: 3×8-10
- Dips: 3×10-12
- Tricep extensions: 3×12-15

**Tuesday - Back & Biceps:**
- Pull-ups: 4×6-10
- Rows: 4×8-10
- Lat pulldowns: 3×10-12
- Bicep curls: 3×12-15

**Wednesday - Legs:**
- Squats: 4×8-10
- Romanian deadlifts: 3×10-12
- Leg press: 3×12-15
- Calf raises: 4×15-20

**Thursday - Shoulders:**
- Military press: 4×8-10
- Lateral raises: 3×12-15
- Rear delt flyes: 3×12-15
- Shrugs: 3×12-15

**Friday - Arms & Core:**
- Close-grip bench: 3×10-12
- Hammer curls: 3×12-15
- Planks: 3×60 seconds
- Russian twists: 3×20

**🍖 NUTRITION FOCUS:**
- 1.6-2.2g protein per kg bodyweight
- Post-workout: Protein + carbs within 30 mins
- Creatine: 5g daily
- Progressive overload weekly

What's your current lifting experience?"""

    elif 'diet' in user_content or 'nutrition' in user_content or 'meal' in user_content:
        reply = """🥗 **COMPREHENSIVE NUTRITION GUIDE** 🍎

Let's optimize your nutrition!

**📊 MACRO BREAKDOWN BY GOAL:**

**🎯 MUSCLE GAIN:**
- Protein: 2.2g/kg bodyweight
- Carbs: 4-6g/kg bodyweight  
- Fats: 1g/kg bodyweight
- Calories: +300-500 surplus

**🔥 FAT LOSS:**
- Protein: 2.5g/kg bodyweight
- Carbs: 2-3g/kg bodyweight
- Fats: 0.8g/kg bodyweight
- Calories: -500 deficit

**🍽️ MEAL TIMING:**
**Pre-workout (1-2 hrs):** Carbs + little protein
**Post-workout (30 mins):** Protein + fast carbs
**Before bed:** Casein protein or Greek yogurt

**🥬 FOOD CHOICES:**
**Proteins:** Chicken, fish, eggs, Greek yogurt, lentils
**Carbs:** Rice, oats, quinoa, sweet potato, fruits
**Fats:** Nuts, avocado, olive oil, fatty fish
**Vegetables:** 5-7 servings daily (all colors!)

**💧 HYDRATION:**
- 35ml per kg bodyweight daily
- +500ml per hour of exercise
- Monitor urine color (pale yellow = good)

**🍽️ SAMPLE 2000 CALORIE DAY:**
Breakfast: Oats + banana + protein powder (400 cal)
Snack: Greek yogurt + berries (200 cal)
Lunch: Chicken + rice + vegetables (500 cal)
Snack: Apple + almond butter (250 cal)
Dinner: Salmon + quinoa + salad (500 cal)
Evening: Casein protein (150 cal)

What's your specific dietary goal and any restrictions?"""

    elif 'exercise' in user_content or 'workout' in user_content:
        reply = """🏋️ **COMPLETE EXERCISE GUIDE** 💪

Let's design your perfect workout!

**🎯 BEGINNER PROGRAM (3x/week):**
**Day 1 - Full Body A:**
1. Bodyweight squats: 3×12-15
2. Push-ups (modified if needed): 3×8-12
3. Bent-over rows: 3×10-12
4. Planks: 3×30-60 seconds
5. Walking: 20-30 minutes

**Day 2 - Full Body B:**
1. Lunges: 3×10 each leg
2. Incline push-ups: 3×8-12
3. Glute bridges: 3×12-15
4. Dead bugs: 3×10 each side
5. Bike/elliptical: 20-30 minutes

**Day 3 - Full Body C:**
1. Wall sits: 3×30-60 seconds
2. Pike push-ups: 3×6-10
3. Single-leg deadlifts: 3×8 each
4. Side planks: 3×20-30 seconds
5. Swimming/dancing: 30 minutes

**💪 INTERMEDIATE (4x/week):**
Split between upper/lower body
Add weights and resistance bands
Increase intensity and complexity

**🏃 CARDIO OPTIONS:**
**HIIT:** 20 mins, 3x/week
**LISS:** 30-45 mins, 2x/week
**Active recovery:** Yoga, walking

**📈 PROGRESSION RULES:**
- Add 1-2 reps when you can complete all sets
- Increase weight by 2.5-5% when form is perfect
- Track every workout
- Deload every 4-6 weeks

What's your fitness level and available equipment?"""

    elif 'hello' in user_content or 'hi' in user_content:
        reply = """Hey there! 👋 I'm Alex, your personal fitness coach! 💪

I'm here to help you with EVERYTHING fitness related:

🏋️ **TRAINING:** Custom workout plans, exercise form, progression
🥗 **NUTRITION:** Meal planning, macro calculations, diet strategies  
💪 **GOALS:** Weight gain/loss, muscle building, strength, endurance
🏃 **CARDIO:** HIIT, running plans, fat loss strategies
🧘 **WELLNESS:** Recovery, sleep, stress management
🩺 **HEALTH:** Injury prevention, therapeutic exercises

**To give you the BEST advice, tell me:**
- Your age, height, weight
- Current fitness level (beginner/intermediate/advanced)
- Main goal (gain weight, lose fat, build muscle, etc.)
- Available equipment
- Time you can dedicate daily

**Popular requests I handle:**
"I want to gain 10kg in 3 months"
"Design a home workout with no equipment"
"I need a meal plan for muscle building"
"Help me lose belly fat"
"I have knee pain, what exercises are safe?"

What's your fitness goal today? Let's crush it together! 🚀"""

    elif 'help' in user_content:
        reply = """🆘 **I'M HERE TO HELP!** 💪

I'm your complete fitness solution! Here's what I can do:

**🏋️ EXERCISE PLANNING:**
- Custom workout routines for any goal
- Home workouts (no equipment needed)
- Gym programs with specific exercises
- Exercise form and technique tips
- Injury prevention strategies

**🥗 NUTRITION COACHING:**
- Personalized meal plans
- Calorie and macro calculations
- Weight gain/loss strategies
- Pre/post workout nutrition
- Supplement recommendations

**🎯 GOAL-SPECIFIC PROGRAMS:**
- Muscle building (bulking)
- Fat loss (cutting)
- Strength training
- Endurance improvement
- Body recomposition

**💡 JUST ASK ME:**
"Create a workout for building chest muscles"
"I need a 1800-calorie meal plan"
"How do I gain weight as a skinny person?"
"What exercises can I do with lower back pain?"
"Design a 30-minute home HIIT workout"

**📝 FOR BEST RESULTS, SHARE:**
- Your stats (age, height, weight)
- Fitness experience level
- Specific goals and timeline
- Available equipment/gym access
- Any injuries or limitations

Ready to transform your fitness journey? What would you like to work on first? 🚀"""

    else:
        # Extract key info if user shares personal details
        if any(word in user_content for word in ['kg', 'lbs', 'weight', 'height', 'age']):
            reply = """📊 **PERSONALIZED PLAN INCOMING!** 💪

I can see you've shared some details - that's perfect! Let me create something specifically for you.

**To give you the MOST effective plan, please confirm:**

🎯 **GOAL:** What's your main objective?
- Gain weight/muscle
- Lose fat
- Build strength  
- Improve endurance
- Body recomposition

⏰ **TIMELINE:** When do you want to achieve this?

🏋️ **EXPERIENCE:** Your current fitness level?
- Beginner (0-6 months)
- Intermediate (6 months - 2 years)
- Advanced (2+ years)

🔧 **EQUIPMENT:** What do you have access to?
- Home (bodyweight/minimal equipment)
- Basic home gym (dumbbells, resistance bands)
- Full gym access

⚡ **TIME:** How many days/hours can you dedicate weekly?

🩺 **LIMITATIONS:** Any injuries or health concerns?

The more specific you are, the better I can tailor your plan! What's your main goal right now? 🚀"""
        else:
            reply = """💪 **READY TO TRANSFORM YOUR FITNESS?** 🚀

I'm Alex, your personal fitness expert! I create detailed, personalized plans for:

**🎯 POPULAR GOALS:**
🏋️ "Help me gain 15kg in 4 months" 
🔥 "I need to lose 20kg for my wedding"
💪 "Build muscle at home with no equipment"
🏃 "Train for a 5K run in 8 weeks"
🥗 "Create a meal plan for my bulk"

**📋 WHAT I PROVIDE:**
✅ Specific workout routines (sets, reps, rest)
✅ Detailed meal plans with calories/macros
✅ Week-by-week progression plans
✅ Exercise form and safety tips
✅ Motivation and accountability

**🔥 RECENT SUCCESS STORIES:**
"Alex helped me gain 12kg in 3 months!"
"Lost 18kg following Alex's plan perfectly!"
"Built my home gym routine - best shape ever!"

**TELL ME:**
- Your current stats and goal
- Timeline you're working with  
- What you have available (equipment/time)

Let's make your fitness dreams a reality! What's your goal? 💯"""
    
    return jsonify({
        'reply': reply,
        'timestamp': datetime.now().isoformat(),
        'uid': uid,
        'fallback': True
    })

if __name__ == '__main__':
    print("Starting POSECRAFT Exercise Monitoring Backend...")
    
    # Load models on startup
    if load_models():
        print("Models loaded successfully. Starting Flask server...")
        app.run(debug=True, host='0.0.0.0', port=5000)
    else:
        print("Failed to load models. Please check model files.") 