import React, { useState, useEffect, useRef } from 'react';
import { supabase } from './supabaseClient';
import './App.css';

const API_BASE_URL = 'http://localhost:8001';

const NS_LABELS = {
  basic_sciences:    'Basic Sciences',
  pharmacology:      'Pharmacology',
  clinical_medicine: 'Clinical Medicine',
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
  const [backendStatus, setBackendStatus] = useState('checking'); // 'online'|'offline'|'partial'|'checking'
  const bottomRef = useRef(null);
  const inputRef  = useRef(null);

  // ── Auth ────────────────────────────────────────────────
  useEffect(() => {
    supabase.auth.getSession().then(({ data: { session } }) => {
      if (session) { setUser(session.user); setToken(session.access_token); }
    });
    const { data: { subscription } } = supabase.auth.onAuthStateChange((_, session) => {
      if (session) { setUser(session.user); setToken(session.access_token); }
      else          { setUser(null); setToken(null); }
    });
    return () => subscription?.unsubscribe();
  }, []);

  useEffect(() => {
    if (user) { checkHealth(); startNew(); }
  }, [user]);

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
    const id = `conv_${Math.random().toString(36).substring(2, 9)}`;
    setConversations(prev => [{
      id,
      title: `Session ${id.substring(5, 9).toUpperCase()}`,
      ts: Date.now()
    }, ...prev]);
    setActiveConvId(id);
    setMessages([]);
    inputRef.current?.focus();
  };

  const selectConv = async (id) => {
    if (id === activeConvId) return;
    setActiveConvId(id);
    setLoading(true);
    try {
      const r = await fetch(`${API_BASE_URL}/history/${id}`, {
        headers: { Authorization: `Bearer ${token || 'mock-token'}` }
      });
      if (r.ok) {
        const { history } = await r.json();
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
    } catch { setMessages([]); }
    finally { setLoading(false); }
  };

  // ── Send ────────────────────────────────────────────────
  const handleSend = async (e) => {
    e.preventDefault();
    const q = inputValue.trim();
    if (!q || loading) return;
    setInputValue('');
    setMessages(prev => [...prev, { role: 'user', content: q }]);
    setLoading(true);
    try {
      // 3-minute timeout — first query loads the embedding model (~1-2 min)
      const r = await fetchWithTimeout(`${API_BASE_URL}/query`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          Authorization: `Bearer ${token || 'mock-token'}`,
        },
        body: JSON.stringify({ question: q, conversation_id: activeConvId }),
      }, 180000);
      if (r.ok) {
        const data = await r.json();
        setMessages(prev => [...prev, {
          role: 'assistant',
          content: data.answer,
          sources: data.sources,
          confidenceScore: data.confidence_score,
          refused: data.refused,
        }]);
      } else {
        const detail = await r.json().catch(() => ({}));
        setMessages(prev => [...prev, {
          role: 'assistant',
          content: `Error ${r.status}: ${detail.detail || 'Backend returned an error.'}`,
          refused: true,
          confidenceScore: 0,
        }]);
      }
    } catch (err) {
      const isTimeout = err.name === 'AbortError';
      setMessages(prev => [...prev, {
        role: 'assistant',
        content: isTimeout
          ? 'Request timed out. The first query loads the AI model (~1-2 minutes). Please try again — it will be faster now.'
          : 'Network error — make sure the FastAPI server is running on port 8000.',
        refused: true,
        confidenceScore: 0,
      }]);
    } finally {
      setLoading(false);
      checkHealth();
    }
  };

  // handle Enter key (Shift+Enter = newline)
  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSend(e);
    }
  };

  // ── Auth actions ────────────────────────────────────────
  const signInGoogle = async () => {
    try {
      await supabase.auth.signInWithOAuth({
        provider: 'google',
        options: { redirectTo: window.location.origin },
      });
    } catch (err) { console.error(err); }
  };

  const bypassAuth = () => {
    setUser({ id: '00000000-0000-0000-0000-000000000000', email: 'dev@medatlas.local', user_metadata: { full_name: 'Dev User' } });
    setToken('mock-token');
  };

  const signOut = async () => {
    if (token === 'mock-token') { setUser(null); setToken(null); }
    else { await supabase.auth.signOut(); }
    setConversations([]); setActiveConvId(''); setMessages([]);
  };

  // ── Helpers ─────────────────────────────────────────────
  const confClass = (s) => s >= 0.8 ? 'high' : s >= 0.65 ? 'medium' : 'low';

  // ── Auth Gate ───────────────────────────────────────────
  if (!user) {
    return (
      <div className="auth-screen">
        <div className="auth-card">
          <div className="auth-logo">
            <div className="auth-logo-mark">M</div>
            <span className="auth-logo-name">MedAtlas</span>
          </div>
          <div className="auth-title">Sign in to continue</div>
          <div className="auth-subtitle">
            Answers grounded in 18 medical textbooks — Harrison's, Robbins, Goodman &amp; Gilman's, and more.
          </div>
          <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
            <button className="btn-auth-google" onClick={signInGoogle}>
              <svg width="16" height="16" viewBox="0 0 24 24">
                <path d="M22.56 12.25c0-.78-.07-1.53-.2-2.25H12v4.26h5.92c-.26 1.37-1.04 2.53-2.21 3.31v2.77h3.57c2.08-1.92 3.28-4.74 3.28-8.09z" fill="#4285F4"/>
                <path d="M12 23c2.97 0 5.46-.98 7.28-2.66l-3.57-2.77c-.98.66-2.23 1.06-3.71 1.06-2.86 0-5.29-1.93-6.16-4.53H2.18v2.84C3.99 20.53 7.7 23 12 23z" fill="#34A853"/>
                <path d="M5.84 14.09c-.22-.66-.35-1.36-.35-2.09s.13-1.43.35-2.09V7.06H2.18C1.43 8.55 1 10.22 1 12s.43 3.45 1.18 4.94l2.85-2.22.81-.63z" fill="#FBBC05"/>
                <path d="M12 5.38c1.62 0 3.06.56 4.21 1.64l3.15-3.15C17.45 2.09 14.97 1 12 1 7.7 1 3.99 3.47 2.18 7.06l3.66 2.84c.87-2.6 3.3-4.52 6.16-4.52z" fill="#EA4335"/>
              </svg>
              Continue with Google
            </button>
            <div className="auth-divider" />
            <button className="btn-auth-dev" onClick={bypassAuth}>
              ⚙ Bypass Auth (Local Dev)
            </button>
          </div>
          <div className="auth-note">
            Authenticated sessions only. Your queries are not stored in external servers during local development.
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
            <div className="logo-mark">M</div>
            <span className="logo-name">MedAtlas</span>
          </div>
          <div className="topbar-divider" />
          <span className="topbar-subtitle">Medical Textbook Knowledge Assistant</span>
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
            <button
              key={conv.id}
              className={`conv-item ${conv.id === activeConvId ? 'active' : ''}`}
              onClick={() => selectConv(conv.id)}
            >
              <span className="conv-dot" />
              {conv.title}
            </button>
          ))}
        </div>

        <div className="sidebar-info">
          <div className="sidebar-info-row">
            <span>User:</span>
            <strong>{user.user_metadata?.full_name || user.email}</strong>
          </div>
          <div className="sidebar-info-row">
            <span>Sources: 18 indexed textbooks</span>
          </div>
          <div className="sidebar-info-row">
            <span>Confidence gate enforced</span>
          </div>
        </div>
      </aside>

      {/* Chat Panel */}
      <main className="chat-panel">
        <div className="messages-area">
          {messages.length === 0 ? (
            <div className="empty-state">
              <div className="empty-icon">🩺</div>
              <h2>Ask a medical question</h2>
              <p>Answers are grounded in authoritative textbooks. Queries with insufficient evidence are refused.</p>
              <div className="suggestions">
                {[
                  ['BIOCHEM', 'Rate-limiting enzyme of glycolysis?'],
                  ['PHARM',   'Mechanism of action of metformin in T2DM?'],
                  ['CLINICAL','Inferior MI — which coronary artery is occluded?'],
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
                <div className={`msg-label ${msg.role === 'user' ? 'user-label' : 'assist-label'}`}>
                  <span className="msg-label-dot" />
                  {msg.role === 'user' ? 'You' : 'MedAtlas'}
                </div>
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
                      <a href="https://www.ncbi.nlm.nih.gov/books/" target="_blank" rel="noopener noreferrer" className="refusal-link">
                        Search NCBI Bookshelf →
                      </a>
                    </div>
                  ) : (
                    <div className="msg-text">{msg.content}</div>
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
                            <span className="citation-score">
                              {(src.relevance_score * 100).toFixed(0)}%
                            </span>
                          </div>
                        ))}
                      </div>
                    </div>
                  )}

                  {/* Confidence */}
                  {msg.role === 'assistant' && msg.confidenceScore != null && (
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
                <span className="loading-text">Searching textbooks… (first query loads AI model, may take 1-2 min)</span>
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
              placeholder="Ask about anatomy, pharmacology, or clinical medicine…"
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
