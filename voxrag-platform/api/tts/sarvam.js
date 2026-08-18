export default async function handler(req, res) {
  // Set CORS headers
  res.setHeader('Access-Control-Allow-Credentials', 'true');
  res.setHeader('Access-Control-Allow-Origin', '*');
  res.setHeader('Access-Control-Allow-Methods', 'GET,OPTIONS,PATCH,DELETE,POST,PUT');
  res.setHeader(
    'Access-Control-Allow-Headers',
    'X-CSRF-Token, X-Requested-With, Accept, Accept-Version, Content-Length, Content-MD5, Content-Type, Date, X-Api-Version'
  );

  if (req.method === 'OPTIONS') {
    res.status(200).end();
    return;
  }

  if (req.method !== 'POST') {
    return res.status(405).json({ error: 'Method not allowed' });
  }

  const { text, speaker = 'priya' } = req.body || {};
  if (!text || typeof text !== 'string' || !text.trim()) {
    return res.status(400).json({ error: 'Valid text required' });
  }

  const apiKey = process.env.SARVAM_API_KEY;
  if (!apiKey) {
    return res.status(500).json({ error: 'SARVAM_API_KEY not configured' });
  }

  // Clean markdown, symbols, and formatting characters for crisp speech
  let cleanText = text
    .replace(/[*#`_~>\[\]\(\)\\]/g, '')
    .replace(/http\S+/g, '')
    .replace(/\s+/g, ' ')
    .trim();

  // Extract first 450 characters (or first 2-3 complete sentences)
  if (cleanText.length > 450) {
    const periodIdx = cleanText.indexOf('.', 280);
    if (periodIdx !== -1 && periodIdx < 450) {
      cleanText = cleanText.slice(0, periodIdx + 1);
    } else {
      cleanText = cleanText.slice(0, 450);
    }
  }

  try {
    const resp = await fetch('https://api.sarvam.ai/text-to-speech', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'api-subscription-key': apiKey
      },
      body: JSON.stringify({
        inputs: [cleanText],
        target_language_code: 'en-IN',
        speaker: speaker,
        model: 'bulbul:v3'
      })
    });

    if (!resp.ok) {
      const errText = await resp.text();
      return res.status(resp.status).json({ error: errText });
    }

    const data = await resp.json();
    if (data && data.audios && data.audios[0]) {
      return res.status(200).json({
        audio_base64: data.audios[0],
        format: 'wav',
        speaker: speaker,
        model: 'bulbul:v3'
      });
    }

    return res.status(500).json({ error: 'No audio returned from Sarvam AI' });
  } catch (err) {
    return res.status(500).json({ error: err.message || 'Error calling Sarvam TTS' });
  }
}
