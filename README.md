# Nachiket Personal Agent

Your personal AI assistant — Tasks, LinkedIn Post Generator, and AI Assistant.

## Setup (10 minutes)

### Step 1 — Add API keys

Open `index.html` and replace these values at the top:

```javascript
const GEMINI_KEY = 'YOUR_GEMINI_API_KEY';         // from Google AI Studio
const GOOGLE_CLIENT_ID = 'YOUR_GOOGLE_CLIENT_ID'; // from Google Cloud
```

- `GEMINI_KEY` — the key copied from Google AI Studio (starts with `AIza...`)
- `GOOGLE_CLIENT_ID` — the OAuth client ID from Google Cloud Console (ends with `.apps.googleusercontent.com`)

### Step 2 — Upload to GitHub

1. Go to github.com
2. Create a new repository: `nachiket-agent`
3. Keep it public
4. Upload all 3 files:
   - `index.html`
   - `vercel.json`
   - `manifest.json`

### Step 3 — Deploy on Vercel

1. Go to vercel.com (log in with GitHub)
2. Click "Add New Project" and select your `nachiket-agent` repo
3. Click "Deploy"
4. In about 2 minutes you get a live URL: `nachiket-agent.vercel.app`

### Step 4 — Update the redirect URI in Google Cloud

In Google Cloud Console, open the Nachiket-Agent OAuth client and add this under Authorized redirect URIs:

```
https://nachiket-agent.vercel.app
```

### Step 5 — Save to your phone's home screen

1. Open your Vercel URL in Chrome on your phone
2. Open the browser menu (3 dots) and tap "Add to Home Screen"
3. Done — you get a dedicated app icon

## Features

- ✅ Tasks — add, complete, and delete tasks with categories
- 💼 LinkedIn — AI-powered Sunday post generator
- 🤖 Assistant — ask anything in Gujarati or English
- 💾 Data is saved locally (localStorage)
- 📱 PWA — can be installed on the home screen
