# HomeValue frontend

Next.js App Router frontend for the FastAPI backend.

```bash
cp .env.example .env.local
npm install
npm run dev
```

Requires Node.js 20.9 or newer. The app provides the Chicago market overview,
property valuation, neighborhood segments, spatial lab, and model research
views. Set `NEXT_PUBLIC_API_URL` when FastAPI is not running at
`http://localhost:8000`.
