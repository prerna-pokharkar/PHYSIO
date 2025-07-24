import React, { createContext, useContext, useEffect, useState } from 'react';
import { 
  User, 
  signInWithPopup, 
  signOut, 
  onAuthStateChanged 
} from 'firebase/auth';
import { auth, googleProvider } from '../firebase';

interface AuthContextType {
  currentUser: User | null;
  loading: boolean;
  signInWithGoogle: () => Promise<void>;
  logout: () => Promise<void>;
}

const AuthContext = createContext<AuthContextType | undefined>(undefined);

export const useAuth = () => {
  const context = useContext(AuthContext);
  if (context === undefined) {
    throw new Error('useAuth must be used within an AuthProvider');
  }
  return context;
};

interface AuthProviderProps {
  children: React.ReactNode;
}

export const AuthProvider: React.FC<AuthProviderProps> = ({ children }) => {
  const [currentUser, setCurrentUser] = useState<User | null>(null);
  const [loading, setLoading] = useState(true);

  const signInWithGoogle = async () => {
    try {
      await signInWithPopup(auth, googleProvider);
    } catch (error) {
      console.error('Error signing in with Google:', error);
      throw error;
    }
  };

  const logout = async () => {
    try {
      // Clear all session storage related to the app
      sessionStorage.clear();
      localStorage.removeItem('disclaimerAccepted');
      
      await signOut(auth);
    } catch (error) {
      console.error('Error signing out:', error);
      throw error;
    }
  };

  useEffect(() => {
    const unsubscribe = onAuthStateChanged(auth, (user) => {
      setCurrentUser(user);
      setLoading(false);
    });

    return unsubscribe;
  }, []);

  // Auto-logout on app close/refresh
  useEffect(() => {
    const handleBeforeUnload = async () => {
      if (currentUser) {
        // Clear session data
        sessionStorage.clear();
        localStorage.removeItem('disclaimerAccepted');
        
        // Force logout - this ensures clean state on next app open
        try {
          await signOut(auth);
        } catch (error) {
          console.error('Auto-logout error:', error);
        }
      }
    };

    const handlePageHide = async () => {
      if (currentUser) {
        // Clear session data when page is hidden (mobile app switching, etc.)
        sessionStorage.clear();
        localStorage.removeItem('disclaimerAccepted');
      }
    };

    // Add event listeners for various close scenarios
    window.addEventListener('beforeunload', handleBeforeUnload);
    window.addEventListener('pagehide', handlePageHide);
    
    // For mobile and modern browsers
    document.addEventListener('visibilitychange', () => {
      if (document.visibilityState === 'hidden' && currentUser) {
        sessionStorage.clear();
        localStorage.removeItem('disclaimerAccepted');
      }
    });

    return () => {
      window.removeEventListener('beforeunload', handleBeforeUnload);
      window.removeEventListener('pagehide', handlePageHide);
    };
  }, [currentUser]);

  const value = {
    currentUser,
    loading,
    signInWithGoogle,
    logout
  };

  return (
    <AuthContext.Provider value={value}>
      {!loading && children}
    </AuthContext.Provider>
  );
}; 