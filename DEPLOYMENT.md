# Railway Deployment Guide

## Prerequisites
1. Railway account: https://railway.app
2. GitHub repository (your ASIS repo)
3. PostgreSQL database on Railway

## Step 1: Connect to ASIS Repository

To connect this project to your ASIS repository, you'll need to provide the repository URL. Then run:

```bash
git remote add origin <YOUR_ASIS_REPO_URL>
git branch -M main
git push -u origin main
```

## Step 2: Deploy to Railway

### Option A: Connect via GitHub (Recommended)
1. Go to https://railway.app/new
2. Select "Deploy from GitHub repo"
3. Choose your ASIS repository
4. Railway will automatically detect this is a Next.js project

### Option B: Deploy via CLI
1. Install Railway CLI: `npm install -g @railway/cli`
2. Login: `railway login`
3. Create project: `railway new`
4. Link to service: `railway link`
5. Deploy: `railway up`

## Step 3: Environment Variables on Railway

Set these environment variables in your Railway project:

### Required Variables:
- `DATABASE_URL` - Will be automatically provided by Railway PostgreSQL
- `NEXTAUTH_SECRET` - Generate with: `openssl rand -base64 32`
- `NEXTAUTH_URL` - Your Railway app URL (e.g., https://your-app.railway.app)

### Optional Variables:
- `NODE_ENV=production`
- `NEXT_PUBLIC_APP_URL` - Same as NEXTAUTH_URL

## Step 4: Database Setup

Railway will automatically:
1. Create a PostgreSQL database
2. Run `prisma db push` to create tables
3. Run the seed script to create default users

## Default Users After Deployment:

### Admin Account:
- Email: `admin@salesportal.com`
- Password: `admin123`

### Agent Account:
- Email: `agent@salesportal.com`
- Password: `agent123`

## Features Available:
- ✅ Sales form submission
- ✅ Admin dashboard
- ✅ User management
- ✅ CSV export with 158 CRM fields
- ✅ Role-based authentication
- ✅ Sales management and deletion
- ✅ Duplicate prevention

## Project Structure:
- **Frontend**: Next.js 14 with TypeScript
- **Backend**: API routes with Prisma ORM
- **Database**: PostgreSQL
- **Authentication**: NextAuth.js
- **Styling**: Tailwind CSS

## Support:
If you encounter any issues during deployment, check:
1. Railway build logs
2. Environment variables are set correctly
3. Database connection is working