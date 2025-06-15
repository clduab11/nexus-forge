import React, { createContext, useContext, useReducer, useEffect, ReactNode } from 'react';
import { nexusForgeApi } from '../services/nexusForgeApi';
import { supabaseService } from '../services/supabaseClient';
import toast from 'react-hot-toast';

interface User {
  id: string;
  email: string;
  name: string;
}

interface AuthState {
  user: User | null;
  isLoading: boolean;
  isAuthenticated: boolean;
  error: string | null;
}

type AuthAction =
  | { type: 'AUTH_START' }
  | { type: 'AUTH_SUCCESS'; payload: User }
  | { type: 'AUTH_FAILURE'; payload: string }
  | { type: 'AUTH_LOGOUT' }
  | { type: 'CLEAR_ERROR' };

const initialState: AuthState = {
  user: null,
  isLoading: true,
  isAuthenticated: false,
  error: null,
};

const authReducer = (state: AuthState, action: AuthAction): AuthState => {
  switch (action.type) {
    case 'AUTH_START':
      return {
        ...state,
        isLoading: true,
        error: null,
      };
    case 'AUTH_SUCCESS':
      return {
        ...state,
        user: action.payload,
        isLoading: false,
        isAuthenticated: true,
        error: null,
      };
    case 'AUTH_FAILURE':
      return {
        ...state,
        user: null,
        isLoading: false,
        isAuthenticated: false,
        error: action.payload,
      };
    case 'AUTH_LOGOUT':
      return {
        ...state,
        user: null,
        isLoading: false,
        isAuthenticated: false,
        error: null,
      };
    case 'CLEAR_ERROR':
      return {
        ...state,
        error: null,
      };
    default:
      return state;
  }
};

interface AuthContextType {
  state: AuthState;
  login: (email: string, password: string) => Promise<void>;
  register: (email: string, password: string, name: string) => Promise<void>;
  logout: () => Promise<void>;
  clearError: () => void;
}

const AuthContext = createContext<AuthContextType | undefined>(undefined);

interface AuthProviderProps {
  children: ReactNode;
}

export const AuthProvider: React.FC<AuthProviderProps> = ({ children }) => {
  const [state, dispatch] = useReducer(authReducer, initialState);

  // Initialize auth state on mount
  useEffect(() => {
    initializeAuth();
  }, []);

  const initializeAuth = async () => {
    dispatch({ type: 'AUTH_START' });

    try {
      // Check for existing token
      const token = localStorage.getItem('access_token');
      if (!token) {
        dispatch({ type: 'AUTH_LOGOUT' });
        return;
      }

      // Verify token with backend
      const user = await supabaseService.getCurrentUser();
      if (user) {
        dispatch({
          type: 'AUTH_SUCCESS',
          payload: {
            id: user.id,
            email: user.email!,
            name: user.user_metadata?.name || user.email!,
          },
        });
      } else {
        // Token invalid, clear storage
        localStorage.removeItem('access_token');
        localStorage.removeItem('refresh_token');
        dispatch({ type: 'AUTH_LOGOUT' });
      }
    } catch (error) {
      console.error('Auth initialization error:', error);
      localStorage.removeItem('access_token');
      localStorage.removeItem('refresh_token');
      dispatch({ type: 'AUTH_LOGOUT' });
    }
  };

  const login = async (email: string, password: string) => {
    dispatch({ type: 'AUTH_START' });

    try {
      // First authenticate with backend
      const authResponse = await nexusForgeApi.login(email, password);
      
      // Store tokens
      localStorage.setItem('access_token', authResponse.access_token);
      localStorage.setItem('refresh_token', authResponse.refresh_token);

      // Then authenticate with Supabase
      await supabaseService.signIn(email, password);

      // Get user info
      const user = await supabaseService.getCurrentUser();
      if (user) {
        dispatch({
          type: 'AUTH_SUCCESS',
          payload: {
            id: user.id,
            email: user.email!,
            name: user.user_metadata?.name || user.email!,
          },
        });
        toast.success('Successfully logged in!');
      } else {
        throw new Error('Failed to get user information');
      }
    } catch (error: any) {
      const errorMessage = error.response?.data?.detail || error.message || 'Login failed';
      dispatch({ type: 'AUTH_FAILURE', payload: errorMessage });
      toast.error(errorMessage);
      throw error;
    }
  };

  const register = async (email: string, password: string, name: string) => {
    dispatch({ type: 'AUTH_START' });

    try {
      // First register with backend
      const authResponse = await nexusForgeApi.register(email, password, name);
      
      // Store tokens
      localStorage.setItem('access_token', authResponse.access_token);
      localStorage.setItem('refresh_token', authResponse.refresh_token);

      // Then register with Supabase
      await supabaseService.signUp(email, password, { name });

      // Get user info
      const user = await supabaseService.getCurrentUser();
      if (user) {
        dispatch({
          type: 'AUTH_SUCCESS',
          payload: {
            id: user.id,
            email: user.email!,
            name: name,
          },
        });
        toast.success('Account created successfully!');
      } else {
        throw new Error('Failed to get user information');
      }
    } catch (error: any) {
      const errorMessage = error.response?.data?.detail || error.message || 'Registration failed';
      dispatch({ type: 'AUTH_FAILURE', payload: errorMessage });
      toast.error(errorMessage);
      throw error;
    }
  };

  const logout = async () => {
    try {
      // Sign out from Supabase
      await supabaseService.signOut();
      
      // Clear local storage
      localStorage.removeItem('access_token');
      localStorage.removeItem('refresh_token');
      
      dispatch({ type: 'AUTH_LOGOUT' });
      toast.success('Successfully logged out');
    } catch (error) {
      console.error('Logout error:', error);
      // Force logout even if there's an error
      localStorage.removeItem('access_token');
      localStorage.removeItem('refresh_token');
      dispatch({ type: 'AUTH_LOGOUT' });
    }
  };

  const clearError = () => {
    dispatch({ type: 'CLEAR_ERROR' });
  };

  const value: AuthContextType = {
    state,
    login,
    register,
    logout,
    clearError,
  };

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>;
};

export const useAuth = () => {
  const context = useContext(AuthContext);
  if (context === undefined) {
    throw new Error('useAuth must be used within an AuthProvider');
  }
  return context;
};

export default AuthContext;