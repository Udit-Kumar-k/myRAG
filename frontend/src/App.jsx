import React, { useState, useEffect, useRef } from 'react';
import { marked } from 'marked';
import DOMPurify from 'dompurify';
import { supabase } from './supabaseClient';
import './App.css';

// Configure marked options for clean legal text formatting
marked.setOptions({
  breaks: true,
  gfm: true,
});

// Backend URL: set VITE_API_BASE_URL in .env (local) or Vercel dashboard (prod).
// Uses relative path if VITE_API_BASE_URL is set to "" (e.g. Dockerfile / HF Spaces),
// falling back to localhost:8001 only when VITE_API_BASE_URL is completely undefined.
const API_BASE_URL = typeof import.meta.env.VITE_API_BASE_URL !== 'undefined'
  ? import.meta.env.VITE_API_BASE_URL
  : 'http://localhost:8001';

const NS_LABELS = {
  criminal: 'Criminal Law',
  cyber:    'Cyber Law',
  consumer: 'Consumer Law',
  banking:  'Banking Law',
  general:  'General',
};

// fetchWithTimeout — prevents UI from hanging forever
const fetchWithTimeout = (url, opts = {}, ms = 120000) => {
  const ctrl = new AbortController();
  const id = setTimeout(() => ctrl.abort(), ms);
  return fetch(url, { ...opts, signal: ctrl.signal })
    .finally(() => clearTimeout(id));
};

function App() {
  const [user, setUser]           = useState(null);
  const [token, setToken]         = useState(null);
  const [conversations, setConversations] = useState([]);
  const [activeConvId, setActiveConvId]   = useState('');
  const [messages, setMessages]   = useState([]);
  const [inputValue, setInputValue] = useState('');
  const [loading, setLoading]     = useState(false);
  const [backendOk, setBackendOk]   = useState(false);
  const [backendStatus, setBackendStatus] = useState('checking');
  const bottomRef = useRef(null);
  const inputRef  = useRef(null);
  // Per-conversation draft: saves the input text when you switch tabs so it's restored
  const draftRef   = useRef({});
  // Abort controller for in-flight /query requests — cancelled on tab switch
  const abortRef   = useRef(null);
  // Sequential session counter so names are Session 1, Session 2 ...
  const sessionCounterRef = useRef(0);

  // ── Auth Form State ───────────────────────────────────
  const [authMode, setAuthMode]         = useState('signin'); // 'signin' | 'signup' | 'forgot'
  const [authEmail, setAuthEmail]       = useState('');
  const [authPassword, setAuthPassword] = useState('');
  const [authConfirm, setAuthConfirm]   = useState('');
  const [authError, setAuthError]       = useState('');
  const [authSuccess, setAuthSuccess]   = useState('');
  const [authLoading, setAuthLoading]   = useState(false);

  // ── Modals & Feedback State ───────────────────────────
  const [showScopeModal, setShowScopeModal]   = useState(false);
  const [flagModalOpen, setFlagModalOpen]     = useState(false);
  const [flagCategory, setFlagCategory]       = useState('wrong_section');
  const [flagComment, setFlagComment]         = useState('');
  const [activeFlagIdx, setActiveFlagIdx]     = useState(null);
  const [feedbackGiven, setFeedbackGiven]     = useState({});

  const submitFeedback = async (msgIndex, rating, category = 'other', comment = '') => {
    const prevUserMsg = messages[msgIndex - 1];
    if (!token) return; // no token, no feedback
    try {
      await fetchWithTimeout(`${API_BASE_URL}/feedback`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${token}`
        },
        body: JSON.stringify({
          conversation_id: activeConvId,
          query: prevUserMsg?.content || '',
          rating,
          category,
          comment
        })
      }, 5000);
      setFeedbackGiven(prev => ({ ...prev, [msgIndex]: rating }));
    } catch (err) {
      console.error('Feedback error:', err);
    }
  };

  const openFlagModal = (msgIndex) => {
    setActiveFlagIdx(msgIndex);
    setFlagCategory('wrong_section');
    setFlagComment('');
    setFlagModalOpen(true);
  };

  const submitFlagFeedback = async () => {
    if (activeFlagIdx !== null) {
      await submitFeedback(activeFlagIdx, 'flag', flagCategory, flagComment);
      setFlagModalOpen(false);
      setActiveFlagIdx(null);
    }
  };

  const initializedUserRef = useRef(null);

  // ── Auth ────────────────────────────────────────────────
  useEffect(() => {
    supabase.auth.getSession().then(({ data: { session } }) => {
      if (session) {
        setUser(prev => (prev?.id === session.user.id ? prev : session.user));
        setToken(session.access_token);
      }
    });
    const { data: { subscription } } = supabase.auth.onAuthStateChange((event, session) => {
      if (session) {
        setUser(prev => (prev?.id === session.user.id ? prev : session.user));
        setToken(session.access_token);
      } else {
        setUser(null);
        setToken(null);
        initializedUserRef.current = null;
      }
    });
    return () => subscription?.unsubscribe();
  }, []);

  const loadHistoryFor = async (id, authToken) => {
    const t = authToken || token;
    if (!id || !t) return;
    setLoading(true);
    try {
      const r = await fetchWithTimeout(`${API_BASE_URL}/history/${id}`, {
        headers: { Authorization: `Bearer ${t}` }
      }, 10000);
      if (r.status === 401 && t !== 'guest-token') {
        await supabase.auth.signOut();
        return;
      }
      if (r.ok) {
        const { history } = await r.json();
        if (!Array.isArray(history)) { setMessages([]); return; }
        const msgs = [];
        history.forEach(item => {
          if (item.question || (item.role === 'user' && item.content)) {
            msgs.push({ role: 'user', content: item.question || item.content });
          }
          if (item.answer || (item.role === 'assistant' && item.content)) {
            msgs.push({
              role: 'assistant',
              content: item.answer || item.content,
              sources: item.sources,
              confidenceScore: item.confidence_score,
              refused: item.refused,
            });
          }
        });
        setMessages(msgs);
      } else {
        setMessages([]);
      }
    } catch {
      setMessages([]);
    } finally {
      setLoading(false);
    }
  };

  const loadConversations = async (authToken) => {
    const t = authToken || token;
    if (!t) return;
    try {
      const r = await fetchWithTimeout(`${API_BASE_URL}/conversations`, {
        headers: { Authorization: `Bearer ${t}` }
      }, 8000);
      if (r.ok) {
        const data = await r.json();
        const serverConvs = data.conversations || [];
        if (serverConvs.length > 0) {
          setConversations(serverConvs);
          const firstId = serverConvs[0].id;
          setActiveConvId(firstId);
          await loadHistoryFor(firstId, t);
          return;
        }
      }
    } catch (e) {
      console.warn('Could not load conversations from server:', e);
    }
    // Fallback: start initial clean session if no conversations returned
    startNew();
  };

  useEffect(() => {
    if (user?.id && token) {
      if (initializedUserRef.current !== user.id) {
        initializedUserRef.current = user.id;
        // Clean any OAuth redirect hash/params from the URL
        if (window.location.hash || window.location.search) {
          window.history.replaceState(null, '', window.location.pathname);
        }
        // Push a sentinel state and lock the back button while logged in
        window.history.pushState({ nyaybot: true }, '', window.location.pathname);
        const handlePopState = () => {
          // Always push forward — prevents navigating away while authenticated
          window.history.pushState({ nyaybot: true }, '', window.location.pathname);
        };
        window.addEventListener('popstate', handlePopState);
        checkHealth();
        // Load user's saved conversations from database
        loadConversations(token);
        return () => window.removeEventListener('popstate', handlePopState);
      }
    } else if (!user?.id) {
      initializedUserRef.current = null;
    }
  }, [user?.id, token]);


  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, loading]);

  // ── Backend ─────────────────────────────────────────────
  const checkHealth = async () => {
    try {
      const r = await fetchWithTimeout(`${API_BASE_URL}/health`, {}, 5000);
      if (r.ok) {
        const data = await r.json();
        const missing = data.missing_namespaces || [];
        setBackendOk(true);
        setBackendStatus(missing.length > 0 ? 'partial' : 'online');
      } else {
        setBackendOk(false);
        setBackendStatus('offline');
      }
    } catch {
      setBackendOk(false);
      setBackendStatus('offline');
    }
  };

  // ── Sessions ────────────────────────────────────────────
  const startNew = () => {
    sessionCounterRef.current += 1;
    const id = `conv_${Math.random().toString(36).substring(2, 9)}`;
    const title = `Session ${sessionCounterRef.current}`;
    setConversations(prev => [{ id, title, ts: Date.now() }, ...prev]);
    // Save current draft before clearing
    draftRef.current[activeConvId] = inputValue;
    setActiveConvId(id);
    setMessages([]);
    setInputValue('');
    inputRef.current?.focus();
  };

  const deleteConv = (e, id) => {
    e.stopPropagation(); // don't trigger selectConv
    if (conversations.length <= 1) return; // Prevent deleting the last remaining session
    const remaining = conversations.filter(c => c.id !== id);
    setConversations(remaining);
    delete draftRef.current[id];
    if (id === activeConvId) {
      // Switch to the first remaining session
      const nextConv = remaining[0];
      if (nextConv) {
        selectConv(nextConv.id);
      } else {
        setMessages([]);
        setActiveConvId('');
        setInputValue('');
      }
    }
    // Delete from database in background
    if (token) {
      fetchWithTimeout(`${API_BASE_URL}/conversations/${id}`, {
        method: 'DELETE',
        headers: { Authorization: `Bearer ${token}` }
      }, 5000).catch(() => {});
    }
  };

  const selectConv = async (id) => {
    if (id === activeConvId || loading) return;
    // Save the current conversation's draft before switching
    draftRef.current[activeConvId] = inputValue;
    setActiveConvId(id);
    setMessages([]);
    // Restore any saved draft for the new conversation
    setInputValue(draftRef.current[id] || '');
    await loadHistoryFor(id, token);
  };

  // ── Send ────────────────────────────────────────────────
  const handleSend = async (e) => {
    e.preventDefault();
    const q = inputValue.trim();
    if (!q || loading) return;
    if (!token) {
      await supabase.auth.signOut();
      return;
    }
    const convIdAtSubmit = activeConvId;
    setInputValue('');
    draftRef.current[convIdAtSubmit] = '';
    setConversations(prev => prev.map(c => {
      if (c.id === convIdAtSubmit && (c.title.startsWith('Session ') || c.title.startsWith('Session_'))) {
        const shortTitle = q.length > 28 ? q.substring(0, 28) + '...' : q;
        return { ...c, title: shortTitle };
      }
      return c;
    }));

    // Append user message and streaming assistant placeholder
    setMessages(prev => [
      ...prev,
      { role: 'user', content: q },
      { role: 'assistant', content: '', sources: [], confidenceScore: null, refused: false, isStreaming: true }
    ]);
    setLoading(true);

    abortRef.current = new AbortController();
    const ctrl = abortRef.current;
    const timeoutId = setTimeout(() => ctrl.abort(), 180000);

    let streamSuccess = false;

    try {
      // Step 1: Try /stream-query first
      const r = await fetch(`${API_BASE_URL}/stream-query`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          Authorization: `Bearer ${token}`,
        },
        body: JSON.stringify({ question: q, conversation_id: convIdAtSubmit }),
        signal: ctrl.signal,
      });

      if (r.status === 401) {
        clearTimeout(timeoutId);
        if (token !== 'guest-token') await supabase.auth.signOut();
        return;
      }

      if (r.ok && r.body) {
        streamSuccess = true;
        const reader = r.body.getReader();
        const decoder = new TextDecoder();
        let buffer = '';

        while (true) {
          const { done, value } = await reader.read();
          if (done) break;
          buffer += decoder.decode(value, { stream: true });
          const lines = buffer.split('\n');
          buffer = lines.pop(); // retain partial line

          for (const line of lines) {
            const trimmed = line.trim();
            if (trimmed.startsWith('data: ')) {
              const jsonStr = trimmed.slice(6).trim();
              if (!jsonStr) continue;
              try {
                const parsed = JSON.parse(jsonStr);
                if (parsed.done) {
                  setMessages(prev => {
                    const updated = [...prev];
                    const last = updated[updated.length - 1];
                    if (last && last.role === 'assistant') {
                      updated[updated.length - 1] = {
                        ...last,
                        content: parsed.answer || last.content,
                        sources: parsed.sources || [],
                        confidenceScore: parsed.confidence_score,
                        refused: parsed.refused || false,
                        isStreaming: false,
                      };
                    }
                    return updated;
                  });
                } else if (parsed.token) {
                  setMessages(prev => {
                    const updated = [...prev];
                    const last = updated[updated.length - 1];
                    if (last && last.role === 'assistant') {
                      updated[updated.length - 1] = {
                        ...last,
                        content: last.content + parsed.token,
                        isStreaming: true,
                      };
                    }
                    return updated;
                  });
                }
              } catch (e) {
                console.error("SSE parse error:", e);
              }
            }
          }
        }
      }
    } catch (err) {
      if (err.name === 'AbortError') {
        clearTimeout(timeoutId);
        return;
      }
      console.warn("Streaming failed or unsupported, falling back to /query:", err);
      streamSuccess = false;
    } finally {
      clearTimeout(timeoutId);
    }

    // Step 2: Fallback to /query if stream failed
    if (!streamSuccess && !ctrl.signal.aborted) {
      try {
        const timeoutId2 = setTimeout(() => ctrl.abort(), 180000);
        const fb = await fetch(`${API_BASE_URL}/query`, {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
            Authorization: `Bearer ${token}`,
          },
          body: JSON.stringify({ question: q, conversation_id: convIdAtSubmit }),
          signal: ctrl.signal,
        }).finally(() => clearTimeout(timeoutId2));

        if (fb.status === 401) {
          if (token !== 'guest-token') await supabase.auth.signOut();
          return;
        }
        if (fb.ok) {
          const data = await fb.json();
          setMessages(prev => {
            const updated = [...prev];
            const last = updated[updated.length - 1];
            if (last && last.role === 'assistant') {
              updated[updated.length - 1] = {
                ...last,
                content: data.answer,
                sources: data.sources || [],
                confidenceScore: data.confidence_score,
                refused: data.refused,
                isStreaming: false,
              };
            }
            return updated;
          });
        } else {
          const detail = await fb.json().catch(() => ({}));
          setMessages(prev => {
            const updated = [...prev];
            const last = updated[updated.length - 1];
            if (last && last.role === 'assistant') {
              updated[updated.length - 1] = {
                ...last,
                content: `Error ${fb.status}: ${detail.detail || 'Backend returned an error.'}`,
                refused: true,
                confidenceScore: 0,
                isStreaming: false,
              };
            }
            return updated;
          });
        }
      } catch (err2) {
        if (err2.name === 'AbortError') return;
        setMessages(prev => {
          const updated = [...prev];
          const last = updated[updated.length - 1];
          if (last && last.role === 'assistant') {
            updated[updated.length - 1] = {
              ...last,
              content: 'Network error — please check your connection or verify the backend service is running.',
              refused: true,
              confidenceScore: 0,
              isStreaming: false,
            };
          }
          return updated;
        });
      }
    }

    setLoading(false);
    checkHealth();
  };

  // handle Enter key (Shift+Enter = newline)
  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSend(e);
    }
  };

  // ── Auth actions ────────────────────────────────────
  const signInGoogle = async () => {
    setAuthError('');
    try {
      await supabase.auth.signInWithOAuth({
        provider: 'google',
        options: { redirectTo: window.location.origin },
      });
    } catch (err) { setAuthError('Google sign-in failed. Please try again.'); }
  };

  const signInEmail = async (e) => {
    e.preventDefault();
    setAuthError(''); setAuthLoading(true);
    try {
      const { error } = await supabase.auth.signInWithPassword({
        email: authEmail, password: authPassword,
      });
      if (error) setAuthError(error.message);
    } catch { setAuthError('Sign-in failed. Check your credentials and try again.'); }
    finally { setAuthLoading(false); }
  };

  const signUpEmail = async (e) => {
    e.preventDefault();
    setAuthError(''); setAuthSuccess('');
    if (authPassword !== authConfirm) { setAuthError('Passwords do not match.'); return; }
    if (authPassword.length < 8) { setAuthError('Password must be at least 8 characters.'); return; }
    setAuthLoading(true);
    try {
      const { error } = await supabase.auth.signUp({
        email: authEmail, password: authPassword,
      });
      if (error) setAuthError(error.message);
      else setAuthSuccess('Account created! Check your email to confirm your address before signing in.');
    } catch { setAuthError('Sign-up failed. Please try again.'); }
    finally { setAuthLoading(false); }
  };

  const sendPasswordReset = async (e) => {
    e.preventDefault();
    setAuthError(''); setAuthSuccess('');
    if (!authEmail) { setAuthError('Enter your email address above.'); return; }
    setAuthLoading(true);
    try {
      const { error } = await supabase.auth.resetPasswordForEmail(authEmail, {
        redirectTo: `${window.location.origin}/reset-password`,
      });
      if (error) setAuthError(error.message);
      else setAuthSuccess('Password reset link sent — check your inbox.');
    } catch { setAuthError('Failed to send reset link. Try again.'); }
    finally { setAuthLoading(false); }
  };

  const enterWithoutLogin = async () => {
    setAuthLoading(true);
    setAuthError('');
    try {
      // 1. Try Supabase Anonymous sign-in (creates real guest session with valid JWT)
      const { data, error } = await supabase.auth.signInAnonymously();
      if (!error && data?.session) {
        setUser(data.session.user);
        setToken(data.session.access_token);
        return;
      }
    } catch (e) {
      console.warn('Supabase anonymous sign-in not configured, using local guest mode:', e);
    }
    // 2. Reliable Guest Mode
    setUser({
      id: '00000000-0000-0000-0000-000000000000',
      email: 'guest@nyaybot.local',
      user_metadata: { full_name: 'Guest User' }
    });
    setToken('guest-token');
    setAuthLoading(false);
  };

  const switchAuthMode = (mode) => {
    setAuthMode(mode); setAuthError(''); setAuthSuccess('');
    setAuthPassword(''); setAuthConfirm('');
  };

  const signOut = async () => {
    initializedUserRef.current = null;
    if (token === 'guest-token' || token === 'mock-token') {
      setUser(null);
      setToken(null);
    } else {
      try { await supabase.auth.signOut(); } catch {}
      setUser(null);
      setToken(null);
    }
    setConversations([]);
    setActiveConvId('');
    setMessages([]);
  };

  // ── Helpers ─────────────────────────────────────────────
  const confClass = (s) => s >= 0.70 ? 'high' : s >= 0.55 ? 'medium' : 'low';

  // ── Auth Gate ───────────────────────────────────────────
  if (!user) {
    return (
      <div className="auth-screen">
        <div className="auth-card">
          {/* Header */}
          <div className="auth-logo">
            <span className="auth-logo-name">NyayBot</span>
          </div>
          <div className="auth-subtitle">
            Grounded in Indian statutory law — BNS, BNSS, BSA, IT Act, Consumer Protection Act, and more.
          </div>

          {/* Tab Switcher */}
          {authMode !== 'forgot' && (
            <div className="auth-tabs">
              <button
                className={`auth-tab ${authMode === 'signin' ? 'active' : ''}`}
                onClick={() => switchAuthMode('signin')}
              >Sign in</button>
              <button
                className={`auth-tab ${authMode === 'signup' ? 'active' : ''}`}
                onClick={() => switchAuthMode('signup')}
              >Create account</button>
            </div>
          )}

          {/* Google Button */}
          {authMode !== 'forgot' && (
            <>
              <button className="btn-auth-google" onClick={signInGoogle} disabled={authLoading}>
                <svg width="16" height="16" viewBox="0 0 24 24">
                  <path d="M22.56 12.25c0-.78-.07-1.53-.2-2.25H12v4.26h5.92c-.26 1.37-1.04 2.53-2.21 3.31v2.77h3.57c2.08-1.92 3.28-4.74 3.28-8.09z" fill="#4285F4"/>
                  <path d="M12 23c2.97 0 5.46-.98 7.28-2.66l-3.57-2.77c-.98.66-2.23 1.06-3.71 1.06-2.86 0-5.29-1.93-6.16-4.53H2.18v2.84C3.99 20.53 7.7 23 12 23z" fill="#34A853"/>
                  <path d="M5.84 14.09c-.22-.66-.35-1.36-.35-2.09s.13-1.43.35-2.09V7.06H2.18C1.43 8.55 1 10.22 1 12s.43 3.45 1.18 4.94l2.85-2.22.81-.63z" fill="#FBBC05"/>
                  <path d="M12 5.38c1.62 0 3.06.56 4.21 1.64l3.15-3.15C17.45 2.09 14.97 1 12 1 7.7 1 3.99 3.47 2.18 7.06l3.66 2.84c.87-2.6 3.3-4.52 6.16-4.52z" fill="#EA4335"/>
                </svg>
                Continue with Google
              </button>
              <div className="auth-or"><span>or</span></div>
            </>
          )}

          {/* Sign In Form */}
          {authMode === 'signin' && (
            <form onSubmit={signInEmail} className="auth-form">
              <div className="auth-field">
                <label className="auth-label" htmlFor="signin-email">Email</label>
                <input
                  id="signin-email"
                  type="email" required
                  className="auth-input"
                  placeholder="you@example.com"
                  value={authEmail}
                  onChange={e => setAuthEmail(e.target.value)}
                />
              </div>
              <div className="auth-field">
                <div className="auth-label-row">
                  <label className="auth-label" htmlFor="signin-password">Password</label>
                  <button type="button" className="auth-link" onClick={() => switchAuthMode('forgot')}>
                    Forgot password?
                  </button>
                </div>
                <input
                  id="signin-password"
                  type="password" required
                  className="auth-input"
                  placeholder="••••••••"
                  value={authPassword}
                  onChange={e => setAuthPassword(e.target.value)}
                />
              </div>
              {authError && <div className="auth-error">{authError}</div>}
              <button type="submit" className="btn-auth-primary" disabled={authLoading}>
                {authLoading ? 'Signing in…' : 'Sign in'}
              </button>
            </form>
          )}

          {/* Sign Up Form */}
          {authMode === 'signup' && (
            <form onSubmit={signUpEmail} className="auth-form">
              <div className="auth-field">
                <label className="auth-label" htmlFor="signup-email">Email</label>
                <input
                  id="signup-email"
                  type="email" required
                  className="auth-input"
                  placeholder="you@example.com"
                  value={authEmail}
                  onChange={e => setAuthEmail(e.target.value)}
                />
              </div>
              <div className="auth-field">
                <label className="auth-label" htmlFor="signup-password">Password</label>
                <input
                  id="signup-password"
                  type="password" required
                  className="auth-input"
                  placeholder="Min. 8 characters"
                  value={authPassword}
                  onChange={e => setAuthPassword(e.target.value)}
                />
              </div>
              <div className="auth-field">
                <label className="auth-label" htmlFor="signup-confirm">Confirm password</label>
                <input
                  id="signup-confirm"
                  type="password" required
                  className="auth-input"
                  placeholder="••••••••"
                  value={authConfirm}
                  onChange={e => setAuthConfirm(e.target.value)}
                />
              </div>
              {authError && <div className="auth-error">{authError}</div>}
              {authSuccess && <div className="auth-success">{authSuccess}</div>}
              <button type="submit" className="btn-auth-primary" disabled={authLoading}>
                {authLoading ? 'Creating account…' : 'Create account'}
              </button>
            </form>
          )}

          {/* Forgot Password Form */}
          {authMode === 'forgot' && (
            <form onSubmit={sendPasswordReset} className="auth-form">
              <button type="button" className="auth-back" onClick={() => switchAuthMode('signin')}>
                ← Back to sign in
              </button>
              <div className="auth-title" style={{ marginTop: '16px' }}>Reset your password</div>
              <div className="auth-subtitle" style={{ marginBottom: '20px' }}>
                Enter your account email and we'll send a reset link.
              </div>
              <div className="auth-field">
                <label className="auth-label" htmlFor="forgot-email">Email</label>
                <input
                  id="forgot-email"
                  type="email" required
                  className="auth-input"
                  placeholder="you@example.com"
                  value={authEmail}
                  onChange={e => setAuthEmail(e.target.value)}
                />
              </div>
              {authError && <div className="auth-error">{authError}</div>}
              {authSuccess && <div className="auth-success">{authSuccess}</div>}
              <button type="submit" className="btn-auth-primary" disabled={authLoading}>
                {authLoading ? 'Sending…' : 'Send reset link'}
              </button>
            </form>
          )}

          {/* Enter without login / Guest access */}
          <div className="auth-dev-row">
            <button
              type="button"
              className="btn-auth-dev"
              onClick={enterWithoutLogin}
              disabled={authLoading}
            >
              Continue without signing in →
            </button>
          </div>
        </div>
      </div>
    );
  }


  // ── Main App ────────────────────────────────────────────
  return (
    <div className="app-shell">
      {/* Topbar */}
      <header className="topbar">
        <div className="topbar-left">
          <div className="logo">
            <span className="logo-name">NyayBot</span>
          </div>
          <div className="topbar-divider" />
          <span className="topbar-subtitle">Indian Legal Awareness Assistant</span>
        </div>
        <div className="topbar-right">
          <div className={`status-pill ${backendStatus}`}>
            <span className={`status-dot ${backendOk ? (backendStatus === 'partial' ? 'partial' : 'online') : 'offline'}`} />
            {backendStatus === 'online'  && 'Engine Online'}
            {backendStatus === 'partial' && 'Partial Index'}
            {backendStatus === 'offline' && 'Engine Offline'}
            {backendStatus === 'checking' && 'Connecting…'}
          </div>
          <button className="btn-ghost" onClick={signOut}>Sign out</button>
        </div>
      </header>

      {/* Top Disclaimer Banner */}
      <div className="disclaimer-banner">
        <span className="disclaimer-icon">⚖️</span>
        <span className="disclaimer-text">
          <strong>Legal Information Notice:</strong> NyayBot provides automated awareness grounded exclusively in indexed Indian statutory enactments. It is not a law firm and does not provide formal legal advice. Always consult a qualified advocate for actionable legal counsel.
        </span>
        <button className="btn-scope-link" onClick={() => setShowScopeModal(true)}>
          Scope & Limitations →
        </button>
      </div>

      {/* Scope Modal */}
      {showScopeModal && (
        <div className="modal-backdrop" onClick={() => setShowScopeModal(false)}>
          <div className="modal-dialog scope-modal" onClick={e => e.stopPropagation()}>
            <div className="modal-header">
              <h3>⚖️ Corpus Scope & System Limitations</h3>
              <button className="btn-close" onClick={() => setShowScopeModal(false)}>✕</button>
            </div>
            <div className="modal-body">
              <div className="scope-section">
                <h4>✅ Grounded & Verified Statutory Domains</h4>
                <ul className="scope-list">
                  <li><strong>Substantive Criminal Law:</strong> Bharatiya Nyaya Sanhita 2023 (BNS)</li>
                  <li><strong>Criminal Procedure:</strong> Bharatiya Nagarik Suraksha Sanhita 2023 (BNSS)</li>
                  <li><strong>Law of Evidence:</strong> Bharatiya Sakshya Adhiniyam 2023 (BSA)</li>
                  <li><strong>Cyber Law:</strong> Information Technology Act, 2000</li>
                  <li><strong>Consumer Protection:</strong> Consumer Protection Act, 2019</li>
                  <li><strong>Commercial Cheque Dishonour:</strong> Negotiable Instruments Act, 1881 (s.138)</li>
                  <li><strong>Civil Contracts & Wages:</strong> Indian Contract Act 1872, Payment of Wages Act 1936</li>
                  <li><strong>Intestate Succession:</strong> Hindu Succession Act 1956 (s.15/s.8)</li>
                  <li><strong>Central Acts:</strong> ~860 central statutes from indiacode.nic.in</li>
                </ul>
              </div>
              <div className="scope-section out-of-scope">
                <h4>❌ Explicitly Out-of-Scope / Untested</h4>
                <ul className="scope-list">
                  <li>State-specific amendments and local tenancy acts (e.g., Delhi/Maharashtra Rent Control)</li>
                  <li>Judicial case law precedents and Supreme Court / High Court case citations</li>
                  <li>Motor accident compensation (MACT claims) and traffic appeals</li>
                  <li>Matrimonial trial procedures and family court applications</li>
                  <li>Taxation (Income Tax, GST, Customs) & Corporate ROC filing compliance</li>
                </ul>
              </div>
            </div>
            <div className="modal-footer">
              <button className="btn-auth-primary" onClick={() => setShowScopeModal(false)}>Acknowledge</button>
            </div>
          </div>
        </div>
      )}

      {/* Flag Modal */}
      {flagModalOpen && (
        <div className="modal-backdrop" onClick={() => setFlagModalOpen(false)}>
          <div className="modal-dialog flag-modal" onClick={e => e.stopPropagation()}>
            <div className="modal-header">
              <h3>🚩 Flag Incorrect Legal Citation / Answer</h3>
              <button className="btn-close" onClick={() => setFlagModalOpen(false)}>✕</button>
            </div>
            <div className="modal-body">
              <p style={{ fontSize: '13px', color: 'var(--text-1)', marginBottom: '12px' }}>
                Help us keep NyayBot accurate. What issue did you observe with this answer?
              </p>
              <div className="auth-field">
                <label className="auth-label">Issue Category</label>
                <select 
                  className="auth-input"
                  value={flagCategory}
                  onChange={e => setFlagCategory(e.target.value)}
                >
                  <option value="wrong_section">Wrong Section Number Cited</option>
                  <option value="outdated_law">Cited Repealed Law (IPC/CrPC instead of BNS/BNSS)</option>
                  <option value="hallucination">Hallucinated / Fabricated Statute</option>
                  <option value="incorrect_advice">Factually Incorrect Legal Information</option>
                  <option value="other">Other Issue</option>
                </select>
              </div>
              <div className="auth-field">
                <label className="auth-label">Optional Comments / Correct Citation</label>
                <textarea
                  className="auth-input"
                  rows={3}
                  placeholder="e.g., The cited section does not apply to non-residential premises..."
                  value={flagComment}
                  onChange={e => setFlagComment(e.target.value)}
                />
              </div>
            </div>
            <div className="modal-footer">
              <button className="btn-ghost" onClick={() => setFlagModalOpen(false)}>Cancel</button>
              <button className="btn-auth-primary" onClick={submitFlagFeedback}>Submit Report</button>
            </div>
          </div>
        </div>
      )}

      {/* Sidebar */}
      <aside className="sidebar">
        <div className="sidebar-top">
          <button className="btn-new-session" onClick={startNew}>
            <svg width="12" height="12" viewBox="0 0 12 12" fill="none">
              <path d="M6 1v10M1 6h10" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round"/>
            </svg>
            New Session
          </button>
        </div>

        <div className="sidebar-section">
          <div className="sidebar-label">Sessions</div>
          {conversations.length === 0 ? (
            <div style={{ padding: '0 12px', fontSize: '12px', color: 'var(--text-2)' }}>
              No sessions yet
            </div>
          ) : conversations.map(conv => (
            <div
              key={conv.id}
              className={`conv-item-row ${conv.id === activeConvId ? 'active' : ''}`}
            >
              <button
                className={`conv-item ${conv.id === activeConvId ? 'active' : ''} ${loading && conv.id !== activeConvId ? 'disabled' : ''}`}
                onClick={() => selectConv(conv.id)}
                title={loading && conv.id !== activeConvId ? 'Wait for current response to finish' : ''}
              >
                <span className="conv-dot" />
                {conv.title}
              </button>
              {conversations.length > 1 && (
                <button
                  className="btn-conv-delete"
                  title="Delete session"
                  onClick={(e) => deleteConv(e, conv.id)}
                >
                  ✕
                </button>
              )}
            </div>
          ))}
        </div>

        <div className="sidebar-info">
          <div className="sidebar-info-row">
            <span>User:</span>
            <strong>{user.user_metadata?.full_name || user.email}</strong>
          </div>
          <div className="sidebar-info-row">
            <span>Engine: Grounded RAG</span>
          </div>
          <div className="sidebar-info-row">
            <button className="btn-sidebar-scope" onClick={() => setShowScopeModal(true)}>
              📋 View Corpus Coverage
            </button>
          </div>
        </div>
      </aside>

      {/* Chat Panel */}
      <main className="chat-panel">
        <div className="messages-area">
          {messages.length === 0 ? (
            <div className="empty-state">
              <div className="empty-icon">⚖️</div>
              <h2>Ask a legal question</h2>
              <p>Answers are grounded in Indian statutory law. Queries with insufficient evidence are refused.</p>
              <div className="suggestions">
                {[
                  ['BNS',      'What does BNS say about cybercrime?'],
                  ['CONSUMER', 'I received a defective product and the seller is refusing a refund'],
                  ['CYBER',    'What are the penalties for hacking under the Information Technology Act?'],
                ].map(([tag, text]) => (
                  <button
                    key={tag}
                    className="suggestion-btn"
                    onClick={() => { setInputValue(text); inputRef.current?.focus(); }}
                  >
                    <span className="suggestion-tag">{tag}</span>
                    {text}
                  </button>
                ))}
              </div>
            </div>
          ) : (
            messages.map((msg, i) => (
              <div key={i} className={`message-row ${msg.role}`}>
                <div className="msg-body">
                  {msg.role === 'assistant' && msg.refused ? (
                    <div className="refusal-block">
                      <div className="refusal-header">
                        <svg width="12" height="12" viewBox="0 0 24 24" fill="none">
                          <path d="M12 9v4M12 17h.01M10.29 3.86L1.82 18a2 2 0 001.71 3h16.94a2 2 0 001.71-3L13.71 3.86a2 2 0 00-3.42 0z" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
                        </svg>
                        Insufficient Grounding
                      </div>
                      <div className="refusal-text">{msg.content}</div>
                      <a href="https://indiacode.nic.in" target="_blank" rel="noopener noreferrer" className="refusal-link">
                        Search India Code →
                      </a>
                    </div>
                  ) : msg.role === 'assistant' ? (
                    <div className="msg-text formatted-markdown">
                      <div dangerouslySetInnerHTML={{ __html: DOMPurify.sanitize(marked.parse(msg.content || ''), { USE_PROFILES: { html: true } }) }} />
                      {msg.isStreaming && <span className="streaming-cursor">▌</span>}
                    </div>
                  ) : (
                    <div className="msg-text user-msg-text">{msg.content}</div>
                  )}

                  {/* Citations */}
                  {msg.role === 'assistant' && msg.sources?.length > 0 && (
                    <div className="citations-block">
                      <div className="citations-title">Sources ({msg.sources.length})</div>
                      <div className="citations-list">
                        {msg.sources.map((src, j) => (
                          <div key={j} className="citation-row">
                            <div className="citation-left">
                              <span className={`ns-tag ns-${src.namespace}`}>
                                {NS_LABELS[src.namespace] || src.namespace}
                              </span>
                              <a
                                href={src.source_url}
                                target="_blank"
                                rel="noopener noreferrer"
                                className="citation-name"
                                title={src.document_name}
                              >
                                {src.document_name}
                              </a>
                            </div>
                            <span className={`citation-score ${confClass(src.relevance_score)}`}>
                              {(src.relevance_score * 100).toFixed(0)}%
                            </span>
                          </div>
                        ))}
                      </div>
                    </div>
                  )}

                  {/* Confidence — only show when answer is grounded in retrieved legal sources */}
                  {msg.role === 'assistant' && msg.confidenceScore != null && msg.sources && msg.sources.length > 0 && (
                    <div className="confidence-block">
                      <span className="confidence-label">Grounding confidence</span>
                      <div className="confidence-track">
                        <div
                          className={`confidence-fill ${confClass(msg.confidenceScore)}`}
                          style={{ width: `${msg.confidenceScore * 100}%` }}
                        />
                      </div>
                      <span className={`confidence-pct ${confClass(msg.confidenceScore)}`}>
                        {(msg.confidenceScore * 100).toFixed(0)}%
                      </span>
                    </div>
                  )}

                  {/* Message Actions & Prominent Disclaimer Footer */}
                  {msg.role === 'assistant' && !msg.refused && (
                    <div className="msg-actions-row">
                      <div className="feedback-buttons">
                        <button
                          className={`btn-action-icon ${feedbackGiven[i] === 'thumbs_up' ? 'active-up' : ''}`}
                          title="Accurate and helpful"
                          onClick={() => submitFeedback(i, 'thumbs_up')}
                        >
                          👍
                        </button>
                        <button
                          className={`btn-action-icon ${feedbackGiven[i] === 'thumbs_down' ? 'active-down' : ''}`}
                          title="Unhelpful"
                          onClick={() => submitFeedback(i, 'thumbs_down')}
                        >
                          👎
                        </button>
                        <button
                          className={`btn-action-icon ${feedbackGiven[i] === 'flag' ? 'active-flag' : ''}`}
                          title="Flag incorrect section / citation"
                          onClick={() => openFlagModal(i)}
                        >
                          🚩 Flag Issue
                        </button>
                        {feedbackGiven[i] && (
                          <span className="feedback-ack">Feedback saved</span>
                        )}
                      </div>
                      <div className="disclaimer-badge">
                        ⚖️ Legal Awareness Only · Not Formal Legal Advice
                      </div>
                    </div>
                  )}
                </div>
              </div>
            ))
          )}

          {loading && (
            <div className="loading-row">
              <div className="loading-body">
                <div className="loading-dots">
                  <span /><span /><span />
                </div>
                <span className="loading-text">Searching legal corpus…</span>
              </div>
            </div>
          )}
          <div ref={bottomRef} />
        </div>

        {/* Input */}
        <div className="input-area">
          <form className="input-form" onSubmit={handleSend}>
            <textarea
              ref={inputRef}
              className="chat-input"
              placeholder="Ask about criminal law, cyber law, consumer rights…"
              value={inputValue}
              onChange={e => setInputValue(e.target.value)}
              onKeyDown={handleKeyDown}
              disabled={loading}
              rows={1}
            />
            <button type="submit" className="btn-send" disabled={loading || !inputValue.trim()}>
              <svg width="16" height="16" viewBox="0 0 24 24" fill="none">
                <path d="M22 2L11 13M22 2L15 22L11 13M11 13L2 9L22 2" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
              </svg>
            </button>
          </form>
          <div className="input-hint">Enter to send · Shift+Enter for newline</div>
        </div>
      </main>
    </div>
  );
}

export default App;
