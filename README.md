# Kwaya — Swahili Choral Score Assistant

[![Kwaya](https://img.shields.io/badge/Kwaya-Swahili_Choir_Assistant-green?style=flat)](https://huggingface.co/spaces/Aniseth/kwaya)

Upload a **MusicXML** choral score and Kwaya will:
- Parse all **4 voices** (Soprano, Alto, Tenor, Bass)
- Display **solfege** and music theory explanations
- **Synthesize** each voice with realistic formant-based singing
- Answer questions through the **AI agent** (powered by DeepSeek V4 via OpenRouter)

Built for Swahili choral teachers and students who need to learn scores quickly without manually decoding every note.

## Stack

- **Backend:** TypeScript + Express (Node.js v22)
- **Database:** SQLite (better-sqlite3)
- **AI Agent:** OpenRouter (DeepSeek V4)
- **Audio:** Pure-TypeScript formant synthesis (no external dependencies)
- **Frontend:** Single-page HTML/CSS with yellow & green choral theme
- **Parser:** MusicXML 4.0 via xml2js

## Quick Start

```bash
# Install
npm install

# Configure API key
cp .env.example .env
# Edit .env — add your OpenRouter API key

# Run
npx tsx src/server.ts
```

Open **http://localhost:8742** in your browser.

## Project Structure

```
kwaya/
├── src/
│   ├── server.ts          # Express routes & API
│   ├── agent.ts           # OpenRouter chat agent
│   ├── db.ts              # SQLite (scores + chats)
│   ├── music-parser.ts    # MusicXML → structured data
│   └── formant-synth.ts   # Pure-TS WAV generation
├── public/
│   ├── index.html         # Chat UI frontend
│   └── style.css          # Yellow & green theme
├── samples/
│   └── mungu-ni-baba.musicxml  # Swahili hymn demo
├── data/                  # SQLite DB (auto-created)
└── uploads/               # Uploaded MusicXML files
```

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/api/status` | Server health & agent status |
| `GET` | `/api/scores` | List all scores |
| `GET` | `/api/scores/:id` | Score detail with notes |
| `POST` | `/api/upload` | Upload MusicXML file (multipart) |
| `POST` | `/api/chat` | Chat with AI agent |
| `POST` | `/api/synthesize` | Generate WAV from note data |
| `DELETE` | `/api/scores/:id` | Delete a score |

## Sample Score

The included **"Mungu Ni Baba"** (a traditional Swahili hymn) demonstrates:
- F major key signature (1 flat — Bb)
- 4-part SATB harmony
- Swahili lyrics per voice
- 8 measures, 4/4 time

## License

MIT