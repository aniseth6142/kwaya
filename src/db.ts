import Database from 'better-sqlite3';
import path from 'path';
import fs from 'fs';

const DB_PATH = path.resolve('data/kwaya.db');

export interface ScoreRow {
  id: number;
  filename: string;
  title: string;
  composer: string;
  key_sig: string;
  time_sig: string;
  voices: string;
  raw_xml: string;
  created_at: string;
}

export interface ChatRow {
  id: number;
  score_id: number;
  role: 'user' | 'assistant' | 'system';
  message: string;
  created_at: string;
}

let db: Database.Database;

export function initDb(): void {
  fs.mkdirSync(path.dirname(DB_PATH), { recursive: true });

  db = new Database(DB_PATH);
  db.pragma('journal_mode = WAL');
  db.pragma('foreign_keys = ON');

  db.exec(`
    CREATE TABLE IF NOT EXISTS scores (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      filename TEXT NOT NULL,
      title TEXT DEFAULT '',
      composer TEXT DEFAULT '',
      key_sig TEXT DEFAULT '',
      time_sig TEXT DEFAULT '',
      voices TEXT DEFAULT '[]',
      raw_xml TEXT DEFAULT '',
      created_at TEXT DEFAULT (datetime('now'))
    );
    CREATE TABLE IF NOT EXISTS chats (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      score_id INTEGER NOT NULL,
      role TEXT NOT NULL CHECK(role IN ('user','assistant','system')),
      message TEXT NOT NULL,
      created_at TEXT DEFAULT (datetime('now')),
      FOREIGN KEY (score_id) REFERENCES scores(id) ON DELETE CASCADE
    );
  `);
}

export function getDb(): Database.Database {
  if (!db) throw new Error('Database not initialised. Call initDb() first.');
  return db;
}

export function insertScore(score: Omit<ScoreRow, 'id' | 'created_at'>): number {
  const s = getDb().prepare(`
    INSERT INTO scores (filename, title, composer, key_sig, time_sig, voices, raw_xml)
    VALUES (@filename, @title, @composer, @key_sig, @time_sig, @voices, @raw_xml)
  `);
  return s.run(score).lastInsertRowid as number;
}

export function getScore(id: number): ScoreRow | null {
  return getDb().prepare('SELECT * FROM scores WHERE id = ?').get(id) as ScoreRow | null;
}

export function listScores(): ScoreRow[] {
  return getDb().prepare('SELECT id, filename, title, composer, key_sig, time_sig, voices, created_at FROM scores ORDER BY created_at DESC').all() as ScoreRow[];
}

export function insertChat(chat: Omit<ChatRow, 'id' | 'created_at'>): number {
  const s = getDb().prepare(`
    INSERT INTO chats (score_id, role, message) VALUES (@score_id, @role, @message)
  `);
  return s.run(chat).lastInsertRowid as number;
}

export function getChatHistory(scoreId: number): ChatRow[] {
  return getDb().prepare('SELECT * FROM chats WHERE score_id = ? ORDER BY id').all(scoreId) as ChatRow[];
}

export function deleteScore(id: number): void {
  getDb().prepare('DELETE FROM chats WHERE score_id = ?').run(id);
  getDb().prepare('DELETE FROM scores WHERE id = ?').run(id);
}

export function closeDb(): void {
  if (db) db.close();
}