# Nachiket Personal Agent

Taro personal AI assistant — Tasks, LinkedIn Post Generator, ane AI Assistant.

## Setup (10 minutes)

### Step 1 — API Keys add karo

`index.html` file kholo ane top ma aa replace karo:

```javascript
const GEMINI_KEY = 'YOUR_GEMINI_API_KEY';         // AI Studio thi
const GOOGLE_CLIENT_ID = 'YOUR_GOOGLE_CLIENT_ID'; // Google Cloud thi
```

Tara actual keys:
- GEMINI_KEY = Google AI Studio thi copy kareli key (AIza...)
- GOOGLE_CLIENT_ID = 676328006759-ufuthu3hb8ge2sno54i98dn3a9oui4fp.apps.googleusercontent.com

### Step 2 — GitHub par upload

1. github.com par ja
2. New repository banavo: `nachiket-agent`
3. Public rakho
4. Badha 3 files upload karo:
   - index.html
   - vercel.json
   - manifest.json

### Step 3 — Vercel par deploy

1. vercel.com par ja (GitHub thi login)
2. "Add New Project" → taro nachiket-agent repo select karo
3. "Deploy" click karo
4. 2 minute ma live URL malse: `nachiket-agent.vercel.app`

### Step 4 — Google Cloud ma Redirect URI update karo

Google Cloud Console → Nachiket-Agent OAuth → Authorized redirect URIs ma add karo:
```
https://nachiket-agent.vercel.app
```

### Step 5 — Mobile par home screen save karo

1. Phone ma Chrome ma taro Vercel URL kholo
2. Browser menu (3 dots) → "Add to Home Screen"
3. Done — dedicated app icon malse!

## Features

- ✅ Tasks — add, complete, delete with categories
- 💼 LinkedIn — AI-powered Sunday post generator  
- 🤖 Assistant — Gujarati/English ma puchho badhu
- 💾 Data locally save rahe (localStorage)
- 📱 PWA — home screen par install thai shake
