import React, { useState, useEffect, useRef } from 'react';
import { supabase } from './supabaseClient';
import './App.css';

const subjectIcons = {
  basic_sciences:   "🔬",
  pharmacology:     "💊",
  clinical_medicine: "🏥",
};

const namespaceLabels = {
  basic_sciences:    "Basic Sciences",
  pharmacology:      "Pharmacology",
  clinical_medicine: "Clinical Medicine",
};

const API_BASE_URL = 'http://localhost:8000';

function App() {
  const [user, setUser] = useState(null);
  const [token, setToken] = useState(null);
  const [conversations, setConversations] = useState([]);
  const [activeConvId, setActiveConvId] = useState('');
  const [messages, setMessages] = useState([]);
  const [inputValue, setInputValue] = useState('');
  const [loading, setLoading] = useState(false);
  const [backendHealthy, setBackendHealthy] = useState(false);
  const messagesEndRef = useRef(null);

  useEffect(() => {
    supabase.auth.getSession().then(({ data: { session } }) => {
      if (session) { setUser(session.user); setToken(session.access_token); }
    });
    const { data: { subscription } } = supabase.auth.onAuthStateChange((event, session) => {
      if (session) { setUser(session.user); setToken(session.access_token); }
      else { setUser(null); setToken(null); }
    });
    return () => subscription?.unsubscribe();
  }, []);

  useEffect(() => {
    if (user) { checkBackendHealth(); startNewConversation(); }
  }, [user]);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, loading]);

  const checkBackendHealth = async () => {
    try {
      const res = await fetch(`${API_BASE_URL}/health`);
      setBackendHealthy(res.ok);
    } catch { setBackendHealthy(false); }
  };

  const startNewConversation = () => {
    const newId = `conv_${Math.random().toString(36).substring(2, 11)}`;
    setConversations(prev => [{
      id: newId,
      title: `Session ${newId.substring(5, 9).toUpperCase()}`,
      created: Date.now()
    }, ...prev]);
    setActiveConvId(newId);
    setMessages([]);
  };

  const selectConversation = async (convId) => {
    setActiveConvId(convId);
    setLoading(true);
    try {
      const res = await fetch(`${API_BASE_URL}/history/${convId}`, {
        headers: { 'Authorization': `Bearer ${token || 'mock-token'}` }
      });
      if (res.ok) {
        const data = await res.json();
        const historyMsgs = [];
        data.history.forEach(item => {
          if (item.question) historyMsgs.push({ role: 'user', content: item.question });
          if (item.answer) historyMsgs.push({
            role: 'assistant', content: item.answer,
            sources: item.sources, confidenceScore: item.confidence_score, refused: item.refused
          });
        });
        setMessages(historyMsgs);
      } else { setMessages([]); }
    } catch (e) { console.error(e); setMessages([]); }
    finally { setLoading(false); }
  };

  const handleSendMessage = async (e) => {
    e.preventDefault();
    if (!inputValue.trim() || loading) return;
    const userText = inputValue;
    setInputValue('');
    setMessages(prev => [...prev, { role: 'user', content: userText }]);
    setLoading(true);
    try {
      const res = await fetch(`${API_BASE_URL}/query`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${token || 'mock-token'}`
        },
        body: JSON.stringify({ question: userText, conversation_id: activeConvId })
      });
      if (res.ok) {
        const data = await res.json();
        setMessages(prev => [...prev, {
          role: 'assistant', content: data.answer,
          sources: data.sources, confidenceScore: data.confidence_score, refused: data.refused
        }]);
      } else {
        setMessages(prev => [...prev, {
          role: 'assistant',
          content: "System error: Failed to connect to MedAtlas backend.",
          refused: true, confidenceScore: 0
        }]);
      }
    } catch (err) {
      setMessages(prev => [...prev, {
        role: 'assistant',
        content: "Network error: Make sure the FastAPI server is running on port 8000.",
        refused: true, confidenceScore: 0
      }]);
    } finally { setLoading(false); checkBackendHealth(); }
  };

  const handleSignInGoogle = async () => {
    try {
      await supabase.auth.signInWithOAuth({
        provider: 'google',
        options: { redirectTo: window.location.origin }
      });
    } catch (err) { console.error(err); }
  };

  const handleBypassAuth = () => {
    setUser({ id: "00000000-0000-0000-0000-000000000000", email: "dev@medatlas.local", user_metadata: { full_name: "Development User" } });
    setToken("mock-token");
  };

  const handleSignOut = async () => {
    if (token === "mock-token") { setUser(null); setToken(null); }
    else { await supabase.auth.signOut(); }
    setConversations([]); setActiveConvId(''); setMessages([]);
  };

  const getConfidenceClass = (score) => {
    if (score >= 0.8) return 'confidence-green';
    if (score >= 0.65) return 'confidence-amber';
    return 'confidence-red';
  };

  // ── Auth Gate ─────────────────────────────────────────────────────────────
  if (!user) {
    return (
      <div style={{
        display: 'flex', alignItems: 'center', justifyContent: 'center',
        minHeight: '100vh', padding: '1.5rem',
        backgroundImage: 'radial-gradient(at 0% 0%, hsla(210, 80%, 50%, 0.08) 0px, transparent 50%), radial-gradient(at 100% 0%, hsla(270, 75%, 65%, 0.08) 0px, transparent 50%)'
      }}>
        <div className="card-glass" style={{ maxWidth: '420px', width: '100%', textAlign: 'center', padding: '2.5rem 2rem' }}>
          <div style={{ fontSize: '3.5rem', marginBottom: '1rem' }}>🩺</div>
          <h1 style={{ fontFamily: 'var(--font-heading)', fontWeight: 800, fontSize: '2rem', marginBottom: '0.5rem', letterSpacing: '-0.02em' }}>
            MedAtlas
          </h1>
          <p style={{ color: 'var(--text-secondary)', fontSize: '0.95rem', marginBottom: '2rem', lineHeight: 1.5 }}>
            Medical Textbook Knowledge Assistant — grounded answers from 18 authoritative textbooks including Harrison's, Robbins, and Goodman & Gilman's.
          </p>
          <div style={{ display: 'flex', flexDirection: 'column', gap: '0.75rem' }}>
            <button className="btn-primary" onClick={handleSignInGoogle}
              style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '0.75rem' }}>
              <svg width="18" height="18" viewBox="0 0 24 24" fill="currentColor">
                <path d="M22.56 12.25c0-.78-.07-1.53-.2-2.25H12v4.26h5.92c-.26 1.37-1.04 2.53-2.21 3.31v2.77h3.57c2.08-1.92 3.28-4.74 3.28-8.09z" fill="#4285F4"/>
                <path d="M12 23c2.97 0 5.46-.98 7.28-2.66l-3.57-2.77c-.98.66-2.23 1.06-3.71 1.06-2.86 0-5.29-1.93-6.16-4.53H2.18v2.84C3.99 20.53 7.7 23 12 23z" fill="#34A853"/>
                <path d="M5.84 14.09c-.22-.66-.35-1.36-.35-2.09s.13-1.43.35-2.09V7.06H2.18C1.43 8.55 1 10.22 1 12s.43 3.45 1.18 4.94l2.85-2.22.81-.63z" fill="#FBBC05"/>
                <path d="M12 5.38c1.62 0 3.06.56 4.21 1.64l3.15-3.15C17.45 2.09 14.97 1 12 1 7.7 1 3.99 3.47 2.18 7.06l3.66 2.84c.87-2.6 3.3-4.52 6.16-4.52z" fill="#EA4335"/>
              </svg>
              Sign In with Google
            </button>
            <button className="conv-item" onClick={handleBypassAuth}
              style={{ padding: '0.75rem', fontWeight: 500, fontSize: '0.85rem', color: 'var(--text-secondary)' }}>
              🔒 Bypass Auth (Local Development)
            </button>
          </div>
        </div>
      </div>
    );
  }

  // ── Main App ─────────────────────────────────────────────────────────────
  return (
    <div className="app-container">
      <header className="header">
        <div className="logo-section">
          <h1>MedAtlas</h1>
          <p>Medical Textbook Knowledge Assistant</p>
        </div>
        <div style={{ display: 'flex', alignItems: 'center', gap: '1rem' }}>
          <div className="status-badge">
            <span className="status-indicator"
              style={{ backgroundColor: backendHealthy ? 'var(--primary)' : 'var(--accent-red)' }} />
            <span>{backendHealthy ? 'Retrieval Engine Online' : 'Engine Offline'}</span>
          </div>
          <button onClick={handleSignOut} className="conv-item"
            style={{ padding: '0.45rem 1rem', fontSize: '0.8rem', border: '1px solid var(--border-color)', borderRadius: 'var(--radius-full)' }}>
            Sign Out
          </button>
        </div>
      </header>

      <aside className="sidebar">
        <button className="btn-primary" onClick={startNewConversation}>
          <span>+</span> New Session
        </button>

        <div className="card-glass">
          <h3>Recent Sessions</h3>
          <div className="conv-list">
            {conversations.map(conv => (
              <button
                key={conv.id}
                className={`conv-item ${conv.id === activeConvId ? 'active' : ''}`}
                onClick={() => selectConversation(conv.id)}
              >
                {conv.title}
              </button>
            ))}
          </div>
        </div>

        <div className="card-glass" style={{ fontSize: '0.85rem', color: 'var(--text-secondary)' }}>
          <h3>MedAtlas Scope</h3>
          <p style={{ marginBottom: '0.5rem' }}>
            Authenticated as: <strong style={{ color: 'var(--text-primary)' }}>{user.user_metadata?.full_name || user.email}</strong>
          </p>
          <p style={{ marginBottom: '0.5rem' }}>
            Answers are grounded in 18 indexed medical textbooks covering basic sciences, pharmacology, and clinical medicine.
          </p>
          <p>Confidence gate calibration is enforced to ensure factual correctness and refuse un-grounded claims.</p>
        </div>
      </aside>

      <main className="card-glass chat-area">
        <div className="messages-container">
          {messages.length === 0 ? (
            <div className="empty-state">
              <div className="empty-state-icon">🩺</div>
              <h3>Medical Textbook Knowledge Assistant</h3>
              <p style={{ marginBottom: '1.5rem' }}>
                Ask questions about anatomy, pharmacology, pathology, or clinical medicine — grounded in authoritative textbooks.
              </p>
              <div style={{ display: 'flex', flexDirection: 'column', gap: '0.5rem', width: '100%', maxWidth: '400px' }}>
                <button className="conv-item" onClick={() => setInputValue("What enzyme catalyzes the rate-limiting step of glycolysis?")}>
                  🔬 Rate-limiting enzyme of glycolysis?
                </button>
                <button className="conv-item" onClick={() => setInputValue("What is the mechanism of action of metformin in type 2 diabetes?")}>
                  💊 Mechanism of action of metformin?
                </button>
                <button className="conv-item" onClick={() => setInputValue("A 55-year-old man presents with crushing substernal chest pain and ST elevation in leads II, III, and aVF. Which coronary artery is most likely occluded?")}>
                  🏥 Inferior MI — which artery is occluded?
                </button>
              </div>
            </div>
          ) : (
            messages.map((msg, index) => (
              <div key={index} className={`message-bubble ${msg.role === 'user' ? 'message-user' : 'message-assistant'}`}>
                <h4>{msg.role === 'user' ? 'Medical Query' : 'Textbook Response'}</h4>
                <div>
                  {msg.role === 'assistant' && msg.refused ? (
                    <div className="refusal-card">
                      <div className="refusal-title">
                        <span>⚠️</span> Insufficient Grounding Evidence
                      </div>
                      <p>{msg.content}</p>
                      <a href="https://www.ncbi.nlm.nih.gov/books/" target="_blank" rel="noopener noreferrer" className="refusal-link">
                        Search NCBI Medical Bookshelf &rarr;
                      </a>
                    </div>
                  ) : (
                    <p style={{ whiteSpace: 'pre-line' }}>{msg.content}</p>
                  )}
                </div>

                {msg.role === 'assistant' && msg.sources && msg.sources.length > 0 && (
                  <div className="citations-section">
                    <h5>Textbook Citations</h5>
                    <div className="citations-grid">
                      {msg.sources.map((src, i) => (
                        <div key={i} className="citation-card">
                          <div className="citation-header">
                            <span className="country-badge">
                              {subjectIcons[src.namespace] || "📚"} {namespaceLabels[src.namespace] || src.namespace}
                            </span>
                            <span className={`namespace-badge namespace-${src.namespace}`}>
                              {namespaceLabels[src.namespace] || src.namespace}
                            </span>
                          </div>
                          <a
                            href={src.source_url}
                            target="_blank"
                            rel="noopener noreferrer"
                            className="citation-title"
                            title={src.document_name}
                          >
                            {src.document_name}
                          </a>
                          <div className="citation-footer">
                            <span>Textbook</span>
                            <span className="match-score">{(src.relevance_score * 100).toFixed(0)}% Match</span>
                          </div>
                        </div>
                      ))}
                    </div>
                  </div>
                )}

                {msg.role === 'assistant' && msg.confidenceScore !== undefined && (
                  <div className="confidence-indicator-container">
                    <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.75rem', color: 'var(--text-muted)' }}>
                      <span>Retrieval Grounding Confidence</span>
                      <span>{(msg.confidenceScore * 100).toFixed(0)}%</span>
                    </div>
                    <div className="confidence-bar-outer">
                      <div
                        className={`confidence-bar-inner ${getConfidenceClass(msg.confidenceScore)}`}
                        style={{ width: `${msg.confidenceScore * 100}%` }}
                      />
                    </div>
                  </div>
                )}
              </div>
            ))
          )}

          {loading && (
            <div className="message-bubble message-assistant" style={{ alignSelf: 'flex-start' }}>
              <h4>Textbook Response</h4>
              <div style={{ display: 'flex', gap: '0.35rem', alignItems: 'center', padding: '0.5rem 0' }}>
                <span style={{ width: '8px', height: '8px', borderRadius: '50%', background: 'var(--primary)', animation: 'pulse 1.2s infinite' }} />
                <span style={{ width: '8px', height: '8px', borderRadius: '50%', background: 'var(--primary)', animation: 'pulse 1.2s infinite 0.2s' }} />
                <span style={{ width: '8px', height: '8px', borderRadius: '50%', background: 'var(--primary)', animation: 'pulse 1.2s infinite 0.4s' }} />
                <span style={{ marginLeft: '0.5rem', color: 'var(--text-secondary)', fontSize: '0.85rem' }}>Retrieving and verifying textbook sources...</span>
              </div>
            </div>
          )}
          <div ref={messagesEndRef} />
        </div>

        <div className="input-area">
          <form onSubmit={handleSendMessage} className="input-wrapper">
            <input
              type="text"
              className="chat-input"
              placeholder="Ask about anatomy, pharmacology, or clinical medicine..."
              value={inputValue}
              onChange={(e) => setInputValue(e.target.value)}
              disabled={loading}
            />
            <button type="submit" className="btn-send" disabled={loading || !inputValue.trim()}>
              <svg width="20" height="20" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                <path d="M22 2L11 13M22 2L15 22L11 13M11 13L2 9L22 2" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
              </svg>
            </button>
          </form>
        </div>
      </main>
    </div>
  );
}

export default App;
