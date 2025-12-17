# 🔗 How to Get Your Deployment URL on Railway

## 📍 Where to Find Your URL

### After Deployment Completes:

1. **Go to your Railway Dashboard**
   - Visit: https://railway.app/dashboard
   - You'll see your project listed

2. **Click on Your Project** (e.g., "aqi-mlops")

3. **Find the "Settings" Tab**
   - Click on **"Settings"** in the left sidebar
   - Scroll down to **"Domains"** section

4. **Your URL Will Be:**
   ```
   https://your-project-name.up.railway.app
   ```
   OR
   ```
   https://your-project-name-production.up.railway.app
   ```

---

## 🎯 Quick Steps:

1. **Railway Dashboard** → Your Project
2. **Settings** → **Domains**
3. **Copy the URL** (starts with `https://`)

---

## 🔧 Custom Domain (Optional)

You can also:
- Add a **custom domain** in Settings → Domains
- Railway provides a **free subdomain** automatically

---

## 📱 Access Your App:

- **Streamlit UI**: `https://your-url.railway.app` (main URL)
- **FastAPI Docs**: `https://your-url.railway.app:8000/docs` (if port 8000 is exposed)

**Note**: Railway typically exposes only one port publicly (the one you set in `$PORT`). 
Your Streamlit app will be on the main URL, and FastAPI runs internally.

---

## 🆘 If You Don't See a URL:

1. Check if deployment is still **building** (wait a few minutes)
2. Check **Deployments** tab for build status
3. Check **Logs** tab for any errors

---

## 💡 Pro Tip:

Railway shows the URL in multiple places:
- ✅ Project overview page
- ✅ Settings → Domains
- ✅ Service details page
- ✅ Deployment logs (at the end)

