import { createClient } from '@supabase/supabase-js';

// Accessing Vite environment variables with configured project defaults as fallback
const supabaseUrl = import.meta.env.VITE_SUPABASE_URL || 'https://smhxjbqwzrnfkmghbvmz.supabase.co';
const supabaseAnonKey = import.meta.env.VITE_SUPABASE_ANON_KEY || 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InNtaHhqYnF3enJuZmttZ2hidm16Iiwicm9sZSI6ImFub24iLCJpYXQiOjE3ODE5NjMwMTIsImV4cCI6MjA5NzUzOTAxMn0.mQmYtAyu5pzUyuDk9J05FCxSLpFsdF6MK30ExLTf7zM';

export const supabase = createClient(supabaseUrl, supabaseAnonKey);

