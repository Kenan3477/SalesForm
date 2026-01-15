#!/bin/bash

# Railway Build Script for Sales Form Portal

echo "🚀 Starting Railway build process..."

# Install dependencies
echo "📦 Installing dependencies..."
npm install

# Generate Prisma client
echo "🔧 Generating Prisma client..."
npx prisma generate

# Run database migrations/push
echo "🗄️ Setting up database..."
npx prisma db push --accept-data-loss

# Seed the database with initial data
echo "🌱 Seeding database..."
npx prisma db seed || npm run db:seed

# Build the Next.js application
echo "🏗️ Building Next.js application..."
npm run build

echo "✅ Build complete!"