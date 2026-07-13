from __future__ import annotations

import json
import sqlite3
import re
from datetime import datetime, timezone
from pathlib import Path
import os
import urllib.request
import urllib.parse
from urllib.parse import urlparse, parse_qs
import markdown
from flask import Flask, request, jsonify, send_from_directory, make_response, session, redirect, render_template_string

try:
    import psycopg2
except ImportError:
    psycopg2 = None

ROOT = Path(__file__).resolve().parent
DB_FILE = ROOT / "regression_notes.sqlite3"

GUIDE_DESCRIPTIONS = {
    "regression": "Complete 10-chapter Zero-to-Research ML textbook integrating basics, OLS, GLMs, Tensors, Autograd, Convex Optimization, SVMs, Trees, PCA, Spectral Clustering, and Neural Networks.",
    "generalizedlinearmodelsmathematicalandpython": "The exponential family, link functions, IRLS, deviance, Poisson, binomial, and Gamma regression.",
    "nexttopicagentbrief": "Self-renewing brief outlining the Zero-to-Research ML curriculum guidelines and upcoming topics.",
    "notesreadme": "Instructions and guidelines for using the local offline SQLite notes server."
}

HUB_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Zeeshan Hayat | Machine Learning Researcher & Engineer</title>
  <meta name="description" content="Personal website of Zeeshan Hayat, Machine Learning Researcher & Engineer. Read from scratch machine learning textbooks, auto-diff frameworks, and optimization guides.">
  <meta name="keywords" content="Zeeshan Hayat, Machine Learning, Deep Learning, autograd, convex optimization, regression guide, neural networks, machine learning engineer">
  <meta name="author" content="Zeeshan Hayat">
  <meta name="robots" content="index, follow">
  <link rel="canonical" href="https://zeehayat.com">
  
  <!-- Open Graph / Facebook -->
  <meta property="og:type" content="website">
  <meta property="og:url" content="https://zeehayat.com/">
  <meta property="og:title" content="Zeeshan Hayat | Machine Learning Researcher & Engineer">
  <meta property="og:description" content="Personal website of Zeeshan Hayat, ML Researcher. Exploring the math and raw hardware optimizations behind autograd, GLMs, and neural networks.">
  <meta property="og:image" content="https://zeehayat.com/og-preview.png">

  <!-- Twitter -->
  <meta property="twitter:card" content="summary_large_image">
  <meta property="twitter:url" content="https://zeehayat.com/">
  <meta property="twitter:title" content="Zeeshan Hayat | Machine Learning Researcher & Engineer">
  <meta property="twitter:description" content="Personal website of Zeeshan Hayat, ML Researcher. Exploring the math and raw hardware optimizations behind autograd, GLMs, and neural networks.">
  <meta property="twitter:image" content="https://zeehayat.com/og-preview.png">

  <!-- Schema.org JSON-LD structured data -->
  <script type="application/ld+json">
  {
    "@context": "https://schema.org",
    "@type": "Person",
    "name": "Zeeshan Hayat",
    "jobTitle": "Machine Learning Engineer & Researcher",
    "url": "https://zeehayat.com",
    "email": "zeenux@gmail.com",
    "description": "Machine learning researcher and engineer specializing in structural foundations, autograd frameworks, and hardware-optimized mathematical representations.",
    "sameAs": []
  }
  </script>

  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&family=Outfit:wght@600;700;800&display=swap" rel="stylesheet">
  <style>
    :root {
      --bg: #030712;
      --surface: rgba(15, 23, 42, 0.7);
      --surface-border: rgba(255, 255, 255, 0.07);
      --surface-hover: rgba(30, 41, 59, 0.85);
      --accent: #06b6d4;
      --accent-gradient: linear-gradient(135deg, #06b6d4 0%, #3b82f6 100%);
      --text: #f3f4f6;
      --text-muted: #9ca3af;
      --font-display: 'Outfit', sans-serif;
      --font-body: 'Inter', sans-serif;
      --container-max: 1100px;
    }
    * { box-sizing: border-box; }
    html { scroll-behavior: smooth; }
    body {
      margin: 0;
      font-family: var(--font-body);
      background: var(--bg);
      color: var(--text);
      line-height: 1.6;
      overflow-x: hidden;
    }
    
    /* Background Glow Effects */
    .glow-container {
      position: absolute;
      top: 0;
      left: 0;
      width: 100vw;
      height: 100vh;
      pointer-events: none;
      z-index: 0;
      overflow: hidden;
    }
    .glow-orb {
      position: absolute;
      width: 500px;
      height: 500px;
      border-radius: 50%;
      background: radial-gradient(circle, rgba(6, 182, 212, 0.1) 0%, rgba(59, 130, 246, 0.05) 50%, transparent 100%);
      filter: blur(80px);
      animation: drift 20s infinite alternate;
    }
    .orb-1 { top: -100px; right: -100px; }
    .orb-2 { bottom: -200px; left: -200px; background: radial-gradient(circle, rgba(59, 130, 246, 0.08) 0%, rgba(6, 182, 212, 0.03) 50%, transparent 100%); }
    
    @keyframes drift {
      0% { transform: translate(0, 0) scale(1); }
      100% { transform: translate(50px, 50px) scale(1.1); }
    }

    /* Navbar */
    header.nav-header {
      position: fixed;
      top: 0;
      left: 0;
      width: 100%;
      z-index: 1000;
      backdrop-filter: blur(12px);
      -webkit-backdrop-filter: blur(12px);
      background: rgba(3, 7, 18, 0.7);
      border-bottom: 1px solid var(--surface-border);
      padding: 16px 24px;
    }
    .nav-container {
      max-width: var(--container-max);
      margin: 0 auto;
      display: flex;
      justify-content: space-between;
      align-items: center;
    }
    .logo {
      font-family: var(--font-display);
      font-size: 20px;
      font-weight: 800;
      background: var(--accent-gradient);
      -webkit-background-clip: text;
      -webkit-text-fill-color: transparent;
      text-decoration: none;
      letter-spacing: -0.5px;
    }
    nav.nav-links {
      display: flex;
      gap: 32px;
    }
    nav.nav-links a {
      color: var(--text-muted);
      text-decoration: none;
      font-size: 14px;
      font-weight: 500;
      transition: color 0.25s ease;
    }
    nav.nav-links a:hover {
      color: var(--text);
    }

    /* Hero Section */
    .hero {
      position: relative;
      min-height: 100vh;
      display: flex;
      align-items: center;
      padding: 120px 24px 60px;
      max-width: var(--container-max);
      margin: 0 auto;
      z-index: 1;
    }
    .hero-content {
      max-width: 650px;
    }
    .hero-pretitle {
      font-family: var(--font-display);
      font-size: 16px;
      color: var(--accent);
      text-transform: uppercase;
      letter-spacing: 2px;
      font-weight: 700;
      margin-bottom: 16px;
    }
    .hero h1 {
      font-family: var(--font-display);
      font-size: clamp(40px, 6vw, 68px);
      font-weight: 800;
      margin: 0 0 20px;
      line-height: 1.1;
      letter-spacing: -1.5px;
      color: #fff;
    }
    .hero p.tagline {
      font-size: clamp(16px, 2.5vw, 20px);
      color: var(--text-muted);
      margin: 0 0 36px;
      line-height: 1.6;
    }
    .hero-actions {
      display: flex;
      gap: 16px;
    }
    .btn {
      padding: 14px 28px;
      font-size: 15px;
      font-weight: 600;
      border-radius: 8px;
      cursor: pointer;
      text-decoration: none;
      transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
      font-family: var(--font-body);
      display: inline-block;
    }
    .btn-primary {
      background: var(--accent-gradient);
      color: #030712;
      border: none;
    }
    .btn-primary:hover {
      transform: translateY(-2px);
      box-shadow: 0 8px 24px rgba(6, 182, 212, 0.25);
    }
    .btn-secondary {
      background: transparent;
      border: 1px solid var(--surface-border);
      color: var(--text);
    }
    .btn-secondary:hover {
      background: rgba(255, 255, 255, 0.03);
      border-color: var(--text-muted);
    }

    /* Section layouts */
    section {
      padding: 100px 24px;
      max-width: var(--container-max);
      margin: 0 auto;
      position: relative;
      z-index: 1;
    }
    .section-title {
      font-family: var(--font-display);
      font-size: 32px;
      font-weight: 800;
      margin: 0 0 16px;
      letter-spacing: -0.5px;
    }
    .section-desc {
      color: var(--text-muted);
      max-width: 600px;
      margin: 0 0 48px;
      font-size: 16px;
    }

    /* About Section */
    .about-grid {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 48px;
      align-items: center;
    }
    .about-text p {
      font-size: 16px;
      line-height: 1.8;
      color: var(--text-muted);
      margin-bottom: 24px;
    }
    .about-stats {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 24px;
    }
    .stat-card {
      background: var(--surface);
      border: 1px solid var(--surface-border);
      border-radius: 12px;
      padding: 24px;
      text-align: center;
      transition: border-color 0.3s ease;
    }
    .stat-card:hover {
      border-color: rgba(6, 182, 212, 0.3);
    }
    .stat-card h3 {
      margin: 0 0 8px;
      font-size: 36px;
      font-family: var(--font-display);
      background: var(--accent-gradient);
      -webkit-background-clip: text;
      -webkit-text-fill-color: transparent;
    }
    .stat-card p {
      margin: 0;
      font-size: 14px;
      color: var(--text-muted);
    }

    /* Grid layout for Cards */
    .grid {
      display: grid;
      grid-template-columns: repeat(auto-fill, minmax(320px, 1fr));
      gap: 24px;
    }
    .card {
      background: var(--surface);
      border: 1px solid var(--surface-border);
      border-radius: 12px;
      padding: 32px;
      display: flex;
      flex-direction: column;
      justify-content: space-between;
      transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
      box-shadow: 0 4px 20px rgba(0,0,0,0.2);
      text-decoration: none;
      color: inherit;
    }
    .card:hover {
      transform: translateY(-5px);
      border-color: var(--accent);
      box-shadow: 0 10px 30px rgba(6, 182, 212, 0.1);
    }
    .card h2 {
      margin: 0 0 12px;
      font-size: 20px;
      font-weight: 700;
      color: #fff;
      font-family: var(--font-display);
    }
    .card p {
      margin: 0 0 24px;
      color: var(--text-muted);
      font-size: 14px;
      flex: 1;
      line-height: 1.6;
    }
    .card-meta {
      display: flex;
      justify-content: space-between;
      align-items: center;
      font-size: 13px;
      border-top: 1px solid rgba(255,255,255,0.06);
      padding-top: 20px;
    }
    .notes-badge {
      background: rgba(6, 182, 212, 0.1);
      color: var(--accent);
      padding: 4px 10px;
      border-radius: 99px;
      font-weight: 600;
    }
    .open-btn {
      color: var(--accent);
      font-weight: 700;
      display: flex;
      align-items: center;
      gap: 6px;
    }
    .open-btn::after {
      content: "→";
      transition: transform 0.2s;
    }
    .card:hover .open-btn::after {
      transform: translateX(4px);
    }

    /* Contact Section Layout */
    .contact-grid {
      display: grid;
      grid-template-columns: 4fr 5fr;
      gap: 48px;
    }
    .contact-info h3 {
      font-family: var(--font-display);
      font-size: 24px;
      margin: 0 0 12px;
    }
    .contact-info p {
      color: var(--text-muted);
      margin-bottom: 32px;
      font-size: 15px;
    }
    .info-item {
      display: flex;
      align-items: center;
      gap: 16px;
      margin-bottom: 24px;
    }
    .info-icon {
      width: 44px;
      height: 44px;
      border-radius: 8px;
      background: rgba(6, 182, 212, 0.08);
      display: flex;
      align-items: center;
      justify-content: center;
      color: var(--accent);
      font-size: 18px;
    }
    .info-details h4 {
      margin: 0 0 2px;
      font-size: 14px;
      color: var(--text-muted);
    }
    .info-details p {
      margin: 0;
      font-size: 16px;
      color: #fff;
    }

    /* Form Styles */
    .contact-form {
      background: var(--surface);
      border: 1px solid var(--surface-border);
      border-radius: 16px;
      padding: 32px;
    }
    .form-group {
      margin-bottom: 20px;
    }
    .form-group label {
      display: block;
      margin-bottom: 8px;
      font-size: 14px;
      font-weight: 550;
      color: var(--text-muted);
    }
    .form-group input,
    .form-group textarea {
      width: 100%;
      background: rgba(255, 255, 255, 0.02);
      border: 1px solid var(--surface-border);
      border-radius: 8px;
      padding: 12px;
      font-family: inherit;
      font-size: 14px;
      color: #fff;
      transition: all 0.3s;
    }
    .form-group input:focus,
    .form-group textarea:focus {
      outline: none;
      border-color: var(--accent);
      background: rgba(255, 255, 255, 0.04);
      box-shadow: 0 0 10px rgba(6, 182, 212, 0.1);
    }
    .submit-btn {
      width: 100%;
      padding: 14px;
      font-weight: 700;
      border: none;
      background: var(--accent-gradient);
      color: #030712;
      border-radius: 8px;
      cursor: pointer;
      font-family: inherit;
      font-size: 15px;
      transition: all 0.3s;
    }
    .submit-btn:hover {
      transform: translateY(-2px);
      box-shadow: 0 8px 20px rgba(6, 182, 212, 0.2);
    }
    .submit-btn:active {
      transform: translateY(0);
    }
    
    .form-status {
      margin-top: 16px;
      padding: 12px;
      border-radius: 8px;
      font-size: 14px;
      display: none;
    }
    .form-status.success {
      background: rgba(16, 185, 129, 0.1);
      border: 1px solid rgba(16, 185, 129, 0.3);
      color: #34d399;
      display: block;
    }
    .form-status.error {
      background: rgba(239, 68, 68, 0.1);
      border: 1px solid rgba(239, 68, 68, 0.3);
      color: #f87171;
      display: block;
    }

    /* Email Reveal button */
    .reveal-email-btn {
      background: none;
      border: none;
      color: var(--accent);
      font-weight: 600;
      cursor: pointer;
      padding: 0;
      font-size: 16px;
      text-decoration: underline;
      display: inline-block;
    }
    .reveal-email-btn:hover {
      color: #3b82f6;
    }

    footer.site-footer {
      text-align: center;
      padding: 48px 24px;
      color: var(--text-muted);
      font-size: 13px;
      border-top: 1px solid var(--surface-border);
      background: rgba(15, 23, 42, 0.3);
    }
    
    @media (max-width: 800px) {
      .about-grid, .contact-grid {
        grid-template-columns: 1fr;
        gap: 36px;
      }
    }
  </style>
</head>
<body>
  <div class="glow-container">
    <div class="glow-orb orb-1"></div>
    <div class="glow-orb orb-2"></div>
  </div>

  <header class="nav-header">
    <div class="nav-container">
      <a href="#" class="logo">Zeeshan Hayat</a>
      <nav class="nav-links">
        <!-- SESSION_LINKS -->
      </nav>
    </div>
  </header>

  <main>
    <section class="hero" id="home">
      <div class="hero-content">
        <div class="hero-pretitle">Welcome to my space</div>
        <h1>Zeeshan Hayat</h1>
        <p class="tagline">Research Analyst & Data Scientist specializing in applying machine learning, MIS development, and large-scale data analytics to drive evidence-based decision-making.</p>
        <div class="hero-actions">
          <a href="#guides" class="btn btn-primary">Study My Guides</a>
          <a href="#contact" class="btn btn-secondary">Get in Touch</a>
        </div>
      </div>
    </section>

    <section id="about">
      <h2 class="section-title">About Me</h2>
      <p class="section-desc">Research Analyst and Data Scientist with expertise in machine learning, MIS development, and social impact analytics.</p>
      
      <div class="about-grid" style="display: grid; grid-template-columns: 1.2fr 1fr; gap: 32px; margin-bottom: 48px;">
        <div style="background: var(--surface); border: 1px solid var(--surface-border); border-radius: 16px; padding: 32px;">
          <h3 style="margin-top: 0; color: var(--accent); font-family: var(--font-display); font-size: 22px; margin-bottom: 16px;">Professional Profile</h3>
          <p style="line-height: 1.6; color: var(--text); margin-bottom: 24px;">
            I specialize in applying <strong>machine learning, MIS development, and large-scale data analytics</strong> to drive evidence-based decision-making in donor-funded programmes (EU, UNFPA, AusAID, PPAF, Govt of KP).
          </p>
          <p style="line-height: 1.6; color: var(--text-muted); margin-bottom: 24px;">
            My work includes designing impact assessments, poverty diagnostics, GBV analytics, and predictive credit scoring models that shape policy, enable public–private partnerships, and improve livelihoods for millions.
          </p>
          <div style="border-top: 1px solid var(--surface-border); padding-top: 20px;">
            <h4 style="margin: 0 0 12px; font-size: 15px; text-transform: uppercase; letter-spacing: 1px; color: var(--text-muted);">Education</h4>
            <ul style="list-style: none; padding: 0; margin: 0; display: flex; flex-direction: column; gap: 8px;">
              <li><strong>MS</strong> — Gandhara University</li>
              <li><strong>Masters in Information Technology</strong> — Gomal University</li>
              <li><strong>Masters in International Relations</strong> — University of Peshawar</li>
            </ul>
          </div>
        </div>

        <div style="background: var(--surface); border: 1px solid var(--surface-border); border-radius: 16px; padding: 32px; display: flex; flex-direction: column; justify-content: space-between;">
          <div>
            <h3 style="margin-top: 0; color: var(--accent); font-family: var(--font-display); font-size: 22px; margin-bottom: 20px;">Experience</h3>
            <div style="display: flex; flex-direction: column; gap: 20px;">
              <div>
                <span style="color: var(--accent); font-size: 12px; font-weight: 600; text-transform: uppercase;">2014 – Present</span>
                <h4 style="margin: 4px 0; font-size: 16px;">Manager IT</h4>
                <p style="margin: 0; font-size: 13px; color: var(--text-muted);">Sarhad Rural Support Programme (SRSP)</p>
              </div>
              <div>
                <span style="color: var(--accent); font-size: 12px; font-weight: 600; text-transform: uppercase;">2005 – 2014</span>
                <h4 style="margin: 4px 0; font-size: 16px;">Manager IT / Program Officer IT</h4>
                <p style="margin: 0; font-size: 13px; color: var(--text-muted);">Sarhad Rural Support Programme (SRSP)</p>
              </div>
              <div>
                <span style="color: var(--accent); font-size: 12px; font-weight: 600; text-transform: uppercase;">Remote Roles</span>
                <h4 style="margin: 4px 0; font-size: 16px;">Analytics Consultant & Technical Lead</h4>
                <p style="margin: 0; font-size: 13px; color: var(--text-muted);">OctoFrost AB (Sweden), Topit Technologies, LocumJobsPk</p>
              </div>
            </div>
          </div>
          
          <div style="border-top: 1px solid var(--surface-border); padding-top: 20px; margin-top: 20px; display: grid; grid-template-columns: 1fr 1fr; gap: 16px;">
            <div class="stat-card" style="padding: 16px; text-align: center; background: rgba(255,255,255,0.01);">
              <h3 style="margin: 0; font-size: 24px; color: var(--accent);">40%</h3>
              <p style="margin: 4px 0 0; font-size: 12px; color: var(--text-muted);">Monitoring Efficiency Gain</p>
            </div>
            <div class="stat-card" style="padding: 16px; text-align: center; background: rgba(255,255,255,0.01);">
              <h3 style="margin: 0; font-size: 24px; color: var(--accent);">3.5B</h3>
              <p style="margin: 4px 0 0; font-size: 12px; color: var(--text-muted);">PKR Loan Model Portfolio</p>
            </div>
          </div>
        </div>
      </div>

      <h3 style="text-align: center; font-family: var(--font-display); font-size: 24px; margin-bottom: 24px;">Key Research Projects & Social Impact</h3>
      <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 24px; margin-bottom: 48px;">
        <div style="background: rgba(15, 23, 42, 0.4); border: 1px solid var(--surface-border); border-radius: 12px; padding: 24px;">
          <h4 style="margin: 0 0 8px; color: var(--text); font-size: 16px;">Poverty Alleviation (Govt of KP)</h4>
          <p style="margin: 0; font-size: 13px; line-height: 1.5; color: var(--text-muted);">
            Developed Poverty Score Card (PSC) software based on Schreiner's tool, coordinating surveys covering <strong>100,000+ households</strong> to guide geographic targeting of poverty interventions.
          </p>
        </div>
        <div style="background: rgba(15, 23, 42, 0.4); border: 1px solid var(--surface-border); border-radius: 12px; padding: 24px;">
          <h4 style="margin: 0 0 8px; color: var(--text); font-size: 16px;">EU PEACE Programme Swat & Kohistan</h4>
          <p style="margin: 0; font-size: 13px; line-height: 1.5; color: var(--text-muted);">
            Designed impact assessment methodology for PKR 1.57B micro-hydro projects using Dedoose Outcome Harvesting, directly measuring benefits for <strong>200,000+ rural households</strong>.
          </p>
        </div>
        <div style="background: rgba(15, 23, 42, 0.4); border: 1px solid var(--surface-border); border-radius: 12px; padding: 24px;">
          <h4 style="margin: 0 0 8px; color: var(--text); font-size: 16px;">Gender-Based Violence (UNFPA)</h4>
          <p style="margin: 0; font-size: 13px; line-height: 1.5; color: var(--text-muted);">
            Developed the MIS & GBV cause analytics for KP's largest program, analyzing trends that enabled support for <strong>311,376 WGFS facility users</strong> and <strong>700,000+ participants</strong>.
          </p>
        </div>
        <div style="background: rgba(15, 23, 42, 0.4); border: 1px solid var(--surface-border); border-radius: 12px; padding: 24px;">
          <h4 style="margin: 0 0 8px; color: var(--text); font-size: 16px;">SRSP Microfinance Credit Model</h4>
          <p style="margin: 0; font-size: 13px; line-height: 1.5; color: var(--text-muted);">
            Designed predictive machine learning-based credit scoring model serving <strong>160,000 clients</strong> with PKR 3.5B loan portfolio, improving repayment rates and expanding marginalized access.
          </p>
        </div>
      </div>

      <div style="background: var(--surface); border: 1px solid var(--surface-border); border-radius: 16px; padding: 32px; text-align: center;">
        <h3 style="margin-top: 0; font-family: var(--font-display); font-size: 20px; margin-bottom: 20px;">Skills & Methodologies</h3>
        <div style="display: flex; flex-wrap: wrap; justify-content: center; gap: 10px;">
          <span style="background: rgba(6,182,212,0.08); border: 1px solid rgba(6,182,212,0.15); color: var(--accent); padding: 6px 14px; border-radius: 99px; font-size: 13px; font-weight: 500;">Poverty Score Card Surveys</span>
          <span style="background: rgba(6,182,212,0.08); border: 1px solid rgba(6,182,212,0.15); color: var(--accent); padding: 6px 14px; border-radius: 99px; font-size: 13px; font-weight: 500;">Impact Assessment Frameworks</span>
          <span style="background: rgba(6,182,212,0.08); border: 1px solid rgba(6,182,212,0.15); color: var(--accent); padding: 6px 14px; border-radius: 99px; font-size: 13px; font-weight: 500;">Outcome Harvesting & Mapping</span>
          <span style="background: rgba(6,182,212,0.08); border: 1px solid rgba(6,182,212,0.15); color: var(--accent); padding: 6px 14px; border-radius: 99px; font-size: 13px; font-weight: 500;">MIS/ERP Development</span>
          <span style="background: rgba(6,182,212,0.08); border: 1px solid rgba(6,182,212,0.15); color: var(--accent); padding: 6px 14px; border-radius: 99px; font-size: 13px; font-weight: 500;">Machine Learning (Scikit-Learn, TensorFlow)</span>
          <span style="background: rgba(6,182,212,0.08); border: 1px solid rgba(6,182,212,0.15); color: var(--accent); padding: 6px 14px; border-radius: 99px; font-size: 13px; font-weight: 500;">Statistical Modeling (Python / R)</span>
          <span style="background: rgba(6,182,212,0.08); border: 1px solid rgba(6,182,212,0.15); color: var(--accent); padding: 6px 14px; border-radius: 99px; font-size: 13px; font-weight: 500;">Data Visualization (Matplotlib, Pandas)</span>
        </div>
      </div>
    </section>

    <section id="simulator">
      <h2 class="section-title">Gradient Descent Simulator</h2>
      <p class="section-desc">Visualize optimization in real-time. Adjust parameters to see how learning rate and noise affect convergence speed and stability.</p>
      
      <div style="background: var(--surface); border: 1px solid var(--surface-border); border-radius: 16px; padding: 32px; display: grid; grid-template-columns: 1.2fr 1fr; gap: 32px;">
        <div style="position: relative; width: 100%; height: 320px; background: rgba(3, 7, 18, 0.4); border-radius: 12px; border: 1px solid var(--surface-border); overflow: hidden;">
          <canvas id="simCanvas" style="display: block; width: 100%; height: 100%;"></canvas>
        </div>
        <div style="display: flex; flex-direction: column; justify-content: space-between;">
          <div>
            <h3 style="margin: 0 0 16px; font-family: var(--font-display); font-size: 20px;">Control Panel</h3>
            <div class="form-group">
              <div style="display: flex; justify-content: space-between; margin-bottom: 8px;">
                <label style="margin: 0; font-size: 14px; font-weight: 550; color: var(--text-muted);">Learning Rate (α)</label>
                <span id="lrVal" style="color: var(--accent); font-weight: 600;">0.050</span>
              </div>
              <input type="range" id="simLr" min="0.001" max="0.300" step="0.001" value="0.050" style="width: 100%; accent-color: var(--accent);">
            </div>
            <div class="form-group" style="margin-top: 16px;">
              <div style="display: flex; justify-content: space-between; margin-bottom: 8px;">
                <label style="margin: 0; font-size: 14px; font-weight: 550; color: var(--text-muted);">Data Noise</label>
                <span id="noiseVal" style="color: var(--accent); font-weight: 600;">15</span>
              </div>
              <input type="range" id="simNoise" min="0" max="40" step="1" value="15" style="width: 100%; accent-color: var(--accent);">
            </div>
            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 16px; margin-top: 24px;">
              <button class="btn btn-secondary" id="btnResetPoints" style="padding: 10px; font-size: 14px;">Reset Data</button>
              <button class="btn btn-primary" id="btnToggleSim" style="padding: 10px; font-size: 14px; color: #030712;">Pause</button>
            </div>
          </div>
          
          <div style="background: rgba(255,255,255,0.02); border: 1px solid var(--surface-border); border-radius: 8px; padding: 16px; margin-top: 16px; display: grid; grid-template-columns: 1fr 1fr; gap: 12px; font-size: 13px;">
            <div>
              <span style="color: var(--text-muted);">Iteration:</span>
              <strong id="simIter" style="color: #fff; float: right;">0</strong>
            </div>
            <div>
              <span style="color: var(--text-muted);">MSE Loss:</span>
              <strong id="simLoss" style="color: #34d399; float: right;">0.0000</strong>
            </div>
            <div>
              <span style="color: var(--text-muted);">Weight (w):</span>
              <strong id="simW" style="color: #fff; float: right;">0.0000</strong>
            </div>
            <div>
              <span style="color: var(--text-muted);">Bias (b):</span>
              <strong id="simB" style="color: #fff; float: right;">0.0000</strong>
            </div>
          </div>
        </div>
      </div>
    </section>

    <section id="guides">
      <h2 class="section-title">Publications & Textbooks</h2>
      <p class="section-desc">A selection of my machine learning textbooks and technical guides. Open any guide to study with local SQLite-powered study notes.</p>
      
      <div class="grid">
        <!-- Cards will be populated dynamically -->
      </div>
    </section>

    <section id="contact">
      <h2 class="section-title">Contact</h2>
      <p class="section-desc">Have a question about my research, guides, or looking to collaborate? Drop me a message below.</p>
      
      <div class="contact-grid">
        <div class="contact-info">
          <h3>Get In Touch</h3>
          <p>Feel free to reach out via the secure contact form, or connect with me directly through my spam-protected email below.</p>
          
          <div class="info-item">
            <div class="info-icon">✉</div>
            <div class="info-details">
              <h4>Email</h4>
              <div id="email-container">
                <button class="reveal-email-btn" onclick="revealEmail()">Click to reveal email</button>
              </div>
            </div>
          </div>
          <div class="info-item">
            <div class="info-icon">📍</div>
            <div class="info-details">
              <h4>Location</h4>
              <p>Remote / Distributed</p>
            </div>
          </div>
        </div>
        
        <div class="contact-form">
          <form id="direct-contact-form">
            <div class="form-group">
              <label for="form-name">Name</label>
              <input type="text" id="form-name" required placeholder="Your name">
            </div>
            <div class="form-group">
              <label for="form-email">Email</label>
              <input type="email" id="form-email" required placeholder="your.email@example.com">
            </div>
            <div class="form-group">
              <label for="form-subject">Subject</label>
              <input type="text" id="form-subject" placeholder="General Inquiry">
            </div>
            <div class="form-group">
              <label for="form-message">Message</label>
              <textarea id="form-message" required rows="5" placeholder="Your message..."></textarea>
            </div>
            <button type="submit" class="submit-btn" id="form-submit-btn">Send Message</button>
            <div class="form-status" id="form-status-msg"></div>
          </form>
        </div>
      </div>
    </section>
  </main>

  <footer class="site-footer">
    &copy; 2026 Zeeshan Hayat • Powered by Local SQLite and Gunicorn
  </footer>

  <script>
    function revealEmail() {
      // Obfuscated email to prevent web scrapers from harvesting it
      const user = "zeenux";
      const domain = "gmail.com";
      const email = user + "@" + domain;
      document.getElementById("email-container").innerHTML = `<a href="mailto:${email}" style="color: var(--accent); font-weight: 600; text-decoration: none;">${email}</a>`;
    }

    document.getElementById("direct-contact-form").addEventListener("submit", async (e) => {
      e.preventDefault();
      const btn = document.getElementById("form-submit-btn");
      const status = document.getElementById("form-status-msg");
      
      const payload = {
        name: document.getElementById("form-name").value.trim(),
        email: document.getElementById("form-email").value.trim(),
        subject: document.getElementById("form-subject").value.trim(),
        message: document.getElementById("form-message").value.trim()
      };
      
      btn.disabled = true;
      btn.textContent = "Sending...";
      status.className = "form-status";
      status.textContent = "";
      
      try {
        const res = await fetch("/api/contact", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(payload)
        });
        
        if (res.ok) {
          status.className = "form-status success";
          status.textContent = "Thank you! Your message has been sent successfully.";
          document.getElementById("direct-contact-form").reset();
        } else {
          const txt = await res.text();
          throw new Error(txt || "Failed to send message");
        }
      } catch (err) {
        status.className = "form-status error";
        status.textContent = `Error: ${err.message}`;
      } finally {
        btn.disabled = false;
        btn.textContent = "Send Message";
      }
    });

    // --- Gradient Descent Simulator JS ---
    const canvas = document.getElementById("simCanvas");
    const ctx = canvas.getContext("2d");
    let points = [];
    let w = 0, b = 0;
    let lr = 0.05;
    let noise = 15;
    let iterations = 0;
    let isRunning = true;
    
    function resizeCanvas() {
      const rect = canvas.getBoundingClientRect();
      canvas.width = rect.width * window.devicePixelRatio;
      canvas.height = rect.height * window.devicePixelRatio;
      ctx.scale(window.devicePixelRatio, window.devicePixelRatio);
    }
    
    function generateData() {
      points = [];
      w = 0;
      b = 0;
      iterations = 0;
      const trueW = 1.5;
      const trueB = 40;
      
      for (let i = 0; i < 35; i++) {
        const x = (Math.random() - 0.5) * 200;
        const err = (Math.random() - 0.5) * noise * 4;
        const y = trueW * x + trueB + err;
        points.push({ x, y });
      }
    }
    
    function stepGradientDescent() {
      if (!points.length) return;
      
      let gradW = 0;
      let gradB = 0;
      let loss = 0;
      const n = points.length;
      
      for (let p of points) {
        const pred = w * p.x + b;
        const diff = pred - p.y;
        gradW += diff * p.x;
        gradB += diff;
        loss += diff * diff;
      }
      
      gradW = (2 / n) * gradW;
      gradB = (2 / n) * gradB;
      loss = loss / n;
      
      w -= lr * gradW;
      b -= lr * gradB;
      
      iterations++;
      
      document.getElementById("simIter").textContent = iterations;
      document.getElementById("simLoss").textContent = loss.toFixed(4);
      document.getElementById("simW").textContent = w.toFixed(4);
      document.getElementById("simB").textContent = b.toFixed(4);
    }
    
    function drawSimulation() {
      ctx.clearRect(0, 0, canvas.clientWidth, canvas.clientHeight);
      
      const width = canvas.clientWidth;
      const height = canvas.clientHeight;
      const centerX = width / 2;
      const centerY = height / 2;
      
      ctx.strokeStyle = "rgba(255,255,255,0.03)";
      ctx.lineWidth = 1;
      for (let x = 0; x < width; x += 40) {
        ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, height); ctx.stroke();
      }
      for (let y = 0; y < height; y += 40) {
        ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(width, y); ctx.stroke();
      }
      
      ctx.strokeStyle = "rgba(255,255,255,0.1)";
      ctx.beginPath(); ctx.moveTo(centerX, 0); ctx.lineTo(centerX, height); ctx.stroke();
      ctx.beginPath(); ctx.moveTo(0, centerY); ctx.lineTo(width, centerY); ctx.stroke();
      
      ctx.fillStyle = "rgba(6, 182, 212, 0.7)";
      for (let p of points) {
        ctx.beginPath();
        const screenX = centerX + p.x;
        const screenY = centerY - p.y;
        ctx.arc(screenX, screenY, 5, 0, Math.PI * 2);
        ctx.fill();
        
        ctx.strokeStyle = "rgba(6, 182, 212, 0.3)";
        ctx.lineWidth = 1;
        ctx.stroke();
      }
      
      ctx.strokeStyle = "#3b82f6";
      ctx.lineWidth = 3;
      ctx.shadowColor = "rgba(59, 130, 246, 0.5)";
      ctx.shadowBlur = 8;
      
      ctx.beginPath();
      const startX = -centerX;
      const startY = w * startX + b;
      ctx.moveTo(centerX + startX, centerY - startY);
      
      const endX = centerX;
      const endY = w * endX + b;
      ctx.lineTo(centerX + endX, centerY - endY);
      ctx.stroke();
      
      ctx.shadowBlur = 0;
    }
    
    function simLoop() {
      if (isRunning) {
        stepGradientDescent();
      }
      drawSimulation();
      requestAnimationFrame(simLoop);
    }
    
    document.getElementById("simLr").addEventListener("input", (e) => {
      lr = parseFloat(e.target.value);
      document.getElementById("lrVal").textContent = lr.toFixed(3);
    });
    
    document.getElementById("simNoise").addEventListener("input", (e) => {
      noise = parseInt(e.target.value);
      document.getElementById("noiseVal").textContent = noise;
      generateData();
    });
    
    document.getElementById("btnResetPoints").addEventListener("click", () => {
      generateData();
    });
    
    document.getElementById("btnToggleSim").addEventListener("click", (e) => {
      isRunning = !isRunning;
      e.target.textContent = isRunning ? "Pause" : "Resume";
      e.target.className = isRunning ? "btn btn-primary" : "btn btn-secondary";
    });
    
    window.addEventListener("resize", () => {
      resizeCanvas();
      generateData();
    });
    
    setTimeout(() => {
      resizeCanvas();
      generateData();
      simLoop();
    }, 100);

    (() => {
      // --- Reading Progress & Resume State Logic ---
      const resumeStyles = `
        .resume-modal-overlay {
          position: fixed;
          top: 0;
          left: 0;
          width: 100vw;
          height: 100vh;
          background: rgba(11, 19, 41, 0.6);
          backdrop-filter: blur(8px);
          -webkit-backdrop-filter: blur(8px);
          display: flex;
          justify-content: center;
          align-items: center;
          z-index: 10000;
          opacity: 0;
          transition: opacity 0.3s ease;
        }
        .resume-modal-overlay.active {
          opacity: 1;
        }
        .resume-modal-container {
          border-radius: 16px;
          padding: 32px;
          width: min(500px, 90vw);
          transform: scale(0.9);
          transition: transform 0.3s cubic-bezier(0.34, 1.56, 0.64, 1);
          font-family: system-ui, -apple-system, sans-serif;
          text-align: center;
        }
        .resume-modal-overlay.active .resume-modal-container {
          transform: scale(1);
        }
        .resume-modal-overlay.theme-hub {
          --bg: rgba(15, 23, 42, 0.95);
          --border: 1px solid rgba(6, 182, 212, 0.3);
          --text: #ffffff;
          --text-muted: #9ca3af;
          --shadow: 0 20px 40px rgba(0, 0, 0, 0.5), 0 0 25px rgba(6, 182, 212, 0.2);
          --primary-bg: #06b6d4;
          --primary-text: #030712;
          --primary-hover: #22d3ee;
          --secondary-bg: rgba(255, 255, 255, 0.05);
          --secondary-border: 1px solid rgba(255, 255, 255, 0.15);
          --secondary-text: #ffffff;
          --secondary-hover: rgba(255, 255, 255, 0.12);
        }
        .resume-modal-container {
          background: var(--bg);
          border: var(--border);
          color: var(--text);
          box-shadow: var(--shadow);
        }
        .resume-modal-title {
          font-size: 24px;
          font-weight: 800;
          margin: 0 0 16px;
          letter-spacing: -0.5px;
        }
        .resume-modal-body {
          font-size: 15px;
          line-height: 1.6;
          margin: 0 0 28px;
          color: var(--text-muted);
        }
        .resume-location-badge {
          display: inline-block;
          margin-top: 12px;
          padding: 6px 12px;
          border-radius: 8px;
          background: rgba(6, 182, 212, 0.1);
          color: #06b6d4;
          font-weight: 600;
          font-size: 13px;
        }
        .resume-modal-actions {
          display: flex;
          gap: 12px;
          justify-content: center;
        }
        .resume-btn {
          padding: 12px 24px;
          font-size: 14px;
          font-weight: 700;
          border-radius: 8px;
          cursor: pointer;
          transition: all 0.2s ease;
          font-family: inherit;
        }
        .resume-btn-primary {
          background: var(--primary-bg);
          color: var(--primary-text);
          border: none;
        }
        .resume-btn-primary:hover {
          background: var(--primary-hover);
          transform: translateY(-2px);
          box-shadow: 0 4px 12px rgba(0,0,0,0.1);
        }
        .resume-btn-primary:active {
          transform: translateY(0);
        }
        .resume-btn-secondary {
          background: var(--secondary-bg);
          border: var(--secondary-border);
          color: var(--secondary-text);
        }
        .resume-btn-secondary:hover {
          background: var(--secondary-hover);
          color: var(--secondary-text);
        }
      `;

      function getGuideDisplayName(key) {
        const names = {
          "regression": "Linear & Logistic Regression Guide",
          "generalizedlinearmodelsmathematicalandpython": "Generalized Linear Models Guide",
          "nexttopicagentbrief": "Next Topic Agent Brief",
          "notesreadme": "Notes Readme Guide"
        };
        return names[key] || key.charAt(0).toUpperCase() + key.slice(1);
      }

      async function initResumePrompt() {
        if (sessionStorage.getItem("resume_prompt_dismissed")) return;

        let state = null;
        try {
          const res = await fetch("/api/reading-state");
          if (res.ok) {
            const data = await res.json();
            state = data.state;
          }
        } catch (err) {
          console.warn("Failed to fetch reading state:", err);
        }

        if (!state) {
          try {
            const local = localStorage.getItem("last_reading_state");
            if (local) {
              state = JSON.parse(local);
            }
          } catch (e) {}
        }

        if (!state) return;

        const styleEl = document.createElement("style");
        styleEl.textContent = resumeStyles;
        document.head.appendChild(styleEl);

        const overlay = document.createElement("div");
        overlay.className = "resume-modal-overlay theme-hub";

        const container = document.createElement("div");
        container.className = "resume-modal-container";

        const titleEl = document.createElement("h3");
        titleEl.className = "resume-modal-title";
        titleEl.textContent = "Resume Reading?";

        const bodyEl = document.createElement("div");
        bodyEl.className = "resume-modal-body";
        
        const guideName = getGuideDisplayName(state.guide);
        const locationText = state.title ? `<br><span class="resume-location-badge">${state.title}</span>` : "";
        bodyEl.innerHTML = `Welcome back! Would you like to resume reading <strong>${guideName}</strong>${locationText}, or start from the beginning?`;

        const actions = document.createElement("div");
        actions.className = "resume-modal-actions";

        const btnResume = document.createElement("button");
        btnResume.className = "resume-btn resume-btn-primary";
        btnResume.textContent = "Resume Reading";
        btnResume.addEventListener("click", () => {
          overlay.classList.remove("active");
          setTimeout(() => overlay.remove(), 300);

          localStorage.setItem("restore_scroll", state.scroll_y);
          localStorage.setItem("restore_scroll_percent", state.scroll_percent);
          window.location.href = state.pathname + (state.anchor || "");
        });

        const btnStart = document.createElement("button");
        btnStart.className = "resume-btn resume-btn-secondary";
        btnStart.textContent = "Start from Beginning";
        btnStart.addEventListener("click", () => {
          sessionStorage.setItem("resume_prompt_dismissed", "true");
          overlay.classList.remove("active");
          setTimeout(() => overlay.remove(), 300);
        });

        actions.append(btnResume, btnStart);
        container.append(titleEl, bodyEl, actions);
        overlay.appendChild(container);
        document.body.appendChild(overlay);

        setTimeout(() => overlay.classList.add("active"), 50);
      }

      window.addEventListener("DOMContentLoaded", initResumePrompt);
    })();
  </script>
</body>
</html>
"""

def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()

def normalize_guide_name(name: str) -> str:
    if name.endswith(".html") or name.endswith(".md"):
        name = name[:-5]
    name = name.lower().replace("_", "").replace("-", "")
    if name.endswith("guide"):
        name = name[:-5]
    return name or "general"

def get_guide_title(filename: str) -> str:
    stem = Path(filename).stem
    if stem == "Regression_Mathematical_and_Python_Guide":
        return "Regression: A Mathematical and Python Guide"
    if stem == "Generalized_Linear_Models_Mathematical_and_Python_Guide":
        return "Generalized Linear Models: A Mathematical and Python Guide"
    if stem == "NEXT_TOPIC_AGENT_BRIEF":
        return "Next Topic Agent Brief"
    return stem.replace("_", " ")

def db_execute(query: str, params: tuple = ()) -> tuple[list[dict], int | None]:
    db_host = os.environ.get("DB_HOST")
    if db_host:
        if not psycopg2:
            raise RuntimeError("psycopg2 is not installed but DB_HOST is configured")
        # PostgreSQL path
        query_pg = query.replace("?", "%s")
        conn = psycopg2.connect(
            host=db_host,
            port=os.environ.get("DB_PORT", "5432"),
            database=os.environ.get("DB_NAME", "ml_book"),
            user=os.environ.get("DB_USER", "postgres"),
            password=os.environ.get("DB_PASSWORD", "postgres_secret_pass")
        )
        try:
            with conn.cursor() as cur:
                cur.execute(query_pg, params)
                rows = []
                if cur.description:
                    colnames = [desc[0] for desc in cur.description]
                    rows = [dict(zip(colnames, row)) for row in cur.fetchall()]
                conn.commit()
                
                lastrowid = None
                if "INSERT" in query.upper():
                    try:
                        cur.execute("SELECT LASTVAL()")
                        lastrowid = cur.fetchone()[0]
                    except Exception:
                        pass
                return rows, lastrowid
        finally:
            conn.close()
    else:
        # SQLite path
        with sqlite3.connect(DB_FILE) as conn:
            conn.row_factory = sqlite3.Row
            cur = conn.execute(query, params)
            rows = []
            if cur.description:
                rows = [dict(row) for row in cur.fetchall()]
            conn.commit()
            return rows, cur.lastrowid

def init_db() -> None:
    db_host = os.environ.get("DB_HOST")
    if db_host:
        if not psycopg2:
            return
        conn = psycopg2.connect(
            host=db_host,
            port=os.environ.get("DB_PORT", "5432"),
            database=os.environ.get("DB_NAME", "ml_book"),
            user=os.environ.get("DB_USER", "postgres"),
            password=os.environ.get("DB_PASSWORD", "postgres_secret_pass")
        )
        try:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    CREATE TABLE IF NOT EXISTS notes (
                        id SERIAL PRIMARY KEY,
                        guide VARCHAR(100) NOT NULL DEFAULT 'regression',
                        section VARCHAR(120) NOT NULL DEFAULT 'General',
                        topic_anchor VARCHAR(160) NOT NULL DEFAULT '',
                        topic_title VARCHAR(240) NOT NULL DEFAULT '',
                        title VARCHAR(240) NOT NULL DEFAULT '',
                        body TEXT NOT NULL DEFAULT '',
                        user_email VARCHAR(255) NOT NULL DEFAULT 'anonymous',
                        created_at VARCHAR(100) NOT NULL,
                        updated_at VARCHAR(100) NOT NULL
                    )
                    """
                )
                cur.execute(
                    """
                    CREATE TABLE IF NOT EXISTS reading_state_v2 (
                        user_email VARCHAR(255) NOT NULL,
                        guide VARCHAR(100) NOT NULL,
                        pathname VARCHAR(255) NOT NULL,
                        anchor VARCHAR(160) NOT NULL DEFAULT '',
                        title VARCHAR(240) NOT NULL DEFAULT '',
                        scroll_y INTEGER NOT NULL,
                        scroll_percent REAL NOT NULL,
                        updated_at VARCHAR(100) NOT NULL,
                        PRIMARY KEY (user_email, guide)
                    )
                    """
                )
                cur.execute(
                    """
                    CREATE TABLE IF NOT EXISTS contact_messages (
                        id SERIAL PRIMARY KEY,
                        name VARCHAR(100) NOT NULL,
                        email VARCHAR(100) NOT NULL,
                        subject VARCHAR(200) NOT NULL,
                        message TEXT NOT NULL,
                        created_at VARCHAR(100) NOT NULL
                    )
                    """
                )
                cur.execute("CREATE INDEX IF NOT EXISTS idx_notes_section ON notes(section)")
                cur.execute("CREATE INDEX IF NOT EXISTS idx_notes_topic ON notes(topic_anchor)")
                cur.execute("CREATE INDEX IF NOT EXISTS idx_notes_guide ON notes(guide)")
                cur.execute("CREATE INDEX IF NOT EXISTS idx_notes_updated_at ON notes(updated_at)")
                cur.execute("CREATE INDEX IF NOT EXISTS idx_notes_user_email ON notes(user_email)")
            conn.commit()
        finally:
            conn.close()
    else:
        with sqlite3.connect(DB_FILE) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS notes (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    guide TEXT NOT NULL DEFAULT 'regression',
                    section TEXT NOT NULL DEFAULT 'General',
                    topic_anchor TEXT NOT NULL DEFAULT '',
                    topic_title TEXT NOT NULL DEFAULT '',
                    title TEXT NOT NULL DEFAULT '',
                    body TEXT NOT NULL DEFAULT '',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS reading_state (
                    guide TEXT PRIMARY KEY,
                    pathname TEXT NOT NULL,
                    anchor TEXT NOT NULL,
                    title TEXT NOT NULL,
                    scroll_y INTEGER NOT NULL,
                    scroll_percent REAL NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS reading_state_v2 (
                    user_email TEXT NOT NULL,
                    guide TEXT NOT NULL,
                    pathname TEXT NOT NULL,
                    anchor TEXT NOT NULL,
                    title TEXT NOT NULL,
                    scroll_y INTEGER NOT NULL,
                    scroll_percent REAL NOT NULL,
                    updated_at TEXT NOT NULL,
                    PRIMARY KEY (user_email, guide)
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS contact_messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL,
                    email TEXT NOT NULL,
                    subject TEXT NOT NULL,
                    message TEXT NOT NULL,
                    created_at TEXT NOT NULL
                )
                """
            )
            columns = {row[1] for row in conn.execute("PRAGMA table_info(notes)").fetchall()}
            if "topic_anchor" not in columns:
                conn.execute("ALTER TABLE notes ADD COLUMN topic_anchor TEXT NOT NULL DEFAULT ''")
            if "topic_title" not in columns:
                conn.execute("ALTER TABLE notes ADD COLUMN topic_title TEXT NOT NULL DEFAULT ''")
            if "guide" not in columns:
                conn.execute("ALTER TABLE notes ADD COLUMN guide TEXT NOT NULL DEFAULT 'regression'")
            if "user_email" not in columns:
                conn.execute("ALTER TABLE notes ADD COLUMN user_email TEXT NOT NULL DEFAULT 'anonymous'")
                
            conn.execute("CREATE INDEX IF NOT EXISTS idx_notes_section ON notes(section)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_notes_topic ON notes(topic_anchor)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_notes_guide ON notes(guide)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_notes_updated_at ON notes(updated_at)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_notes_user_email ON notes(user_email)")

def migrate_sqlite_to_postgres() -> None:
    db_host = os.environ.get("DB_HOST")
    if not db_host or not psycopg2:
        return
        
    sqlite_file = ROOT / "regression_notes.sqlite3"
    if not sqlite_file.exists():
        return
        
    try:
        pg_conn = psycopg2.connect(
            host=db_host,
            port=os.environ.get("DB_PORT", "5432"),
            database=os.environ.get("DB_NAME", "ml_book"),
            user=os.environ.get("DB_USER", "postgres"),
            password=os.environ.get("DB_PASSWORD", "postgres_secret_pass")
        )
        try:
            with pg_conn.cursor() as cur:
                cur.execute("SELECT COUNT(*) FROM notes")
                count = cur.fetchone()[0]
                if count > 0:
                    return
        except Exception:
            return
        finally:
            pg_conn.close()
            
        print("Starting SQLite to PostgreSQL migration...")
        lite_conn = sqlite3.connect(sqlite_file)
        lite_conn.row_factory = sqlite3.Row
        
        pg_conn = psycopg2.connect(
            host=db_host,
            port=os.environ.get("DB_PORT", "5432"),
            database=os.environ.get("DB_NAME", "ml_book"),
            user=os.environ.get("DB_USER", "postgres"),
            password=os.environ.get("DB_PASSWORD", "postgres_secret_pass")
        )
        
        try:
            lite_notes = lite_conn.execute("SELECT * FROM notes").fetchall()
            with pg_conn.cursor() as cur:
                for note in lite_notes:
                    cur.execute(
                        """
                        INSERT INTO notes (id, guide, section, topic_anchor, topic_title, title, body, user_email, created_at, updated_at)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                        ON CONFLICT (id) DO NOTHING
                        """,
                        (
                            note["id"], note["guide"], note["section"], note["topic_anchor"],
                            note["topic_title"], note["title"], note["body"],
                            note.get("user_email", "anonymous"), note["created_at"], note["updated_at"]
                        )
                    )
            print(f"Migrated {len(lite_notes)} notes to PostgreSQL.")
            
            lite_tables = {row[0] for row in lite_conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()}
            states_migrated = 0
            with pg_conn.cursor() as cur:
                if "reading_state_v2" in lite_tables:
                    lite_states = lite_conn.execute("SELECT * FROM reading_state_v2").fetchall()
                    for state in lite_states:
                        cur.execute(
                            """
                            INSERT INTO reading_state_v2 (user_email, guide, pathname, anchor, title, scroll_y, scroll_percent, updated_at)
                            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                            ON CONFLICT (user_email, guide) DO NOTHING
                            """,
                            (
                                state["user_email"], state["guide"], state["pathname"], state["anchor"],
                                state["title"], state["scroll_y"], state["scroll_percent"], state["updated_at"]
                            )
                        )
                    states_migrated = len(lite_states)
                elif "reading_state" in lite_tables:
                    lite_states = lite_conn.execute("SELECT * FROM reading_state").fetchall()
                    for state in lite_states:
                        cur.execute(
                            """
                            INSERT INTO reading_state_v2 (user_email, guide, pathname, anchor, title, scroll_y, scroll_percent, updated_at)
                            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                            ON CONFLICT (user_email, guide) DO NOTHING
                            """,
                            (
                                "anonymous", state["guide"], state["pathname"], state["anchor"],
                                state["title"], state["scroll_y"], state["scroll_percent"], state["updated_at"]
                            )
                        )
                    states_migrated = len(lite_states)
            print(f"Migrated {states_migrated} reading records to PostgreSQL.")
            
            if "contact_messages" in lite_tables:
                lite_msgs = lite_conn.execute("SELECT * FROM contact_messages").fetchall()
                with pg_conn.cursor() as cur:
                    for msg in lite_msgs:
                        cur.execute(
                            """
                            INSERT INTO contact_messages (id, name, email, subject, message, created_at)
                            VALUES (%s, %s, %s, %s, %s, %s)
                            ON CONFLICT (id) DO NOTHING
                            """,
                            (
                                msg["id"], msg["name"], msg["email"], msg["subject"], msg["message"], msg["created_at"]
                            )
                        )
                print(f"Migrated {len(lite_msgs)} contact messages to PostgreSQL.")
            pg_conn.commit()
        except Exception as e:
            pg_conn.rollback()
            print("Failed to run SQLite to PostgreSQL migration:", e)
        finally:
            lite_conn.close()
            pg_conn.close()
    except Exception as e:
        print("Database connection error during migration:", e)

def find_markdown_file(stem: str) -> Path | None:
    norm = normalize_guide_name(stem)
    for f in ROOT.glob("*.md"):
        f_norm = normalize_guide_name(f.name)
        if f_norm == norm:
            return f
    for f in ROOT.glob("*.md"):
        f_norm = normalize_guide_name(f.name)
        if f_norm.startswith(norm) or norm.startswith(f_norm) or f_norm in norm or norm in f_norm:
            return f
    return None

def get_note_counts() -> dict[str, int]:
    counts = {"regression": 0}
    for f in ROOT.glob("*.md"):
        counts[normalize_guide_name(f.name)] = 0
    try:
        user_email = get_current_user_email()
        rows, _ = db_execute(
            "SELECT guide, COUNT(*) as cnt FROM notes WHERE user_email = ? GROUP BY guide",
            (user_email,)
        )
        for row in rows:
            counts[normalize_guide_name(row["guide"])] = row["cnt"]
    except Exception as e:
        print("Error getting note counts:", e)
    return counts

def render_markdown_guide(md_file: Path) -> str:
    with open(md_file, "r", encoding="utf-8") as f:
        text = f.read()

    md = markdown.Markdown(extensions=["fenced_code", "tables", "toc"])
    html_body = md.convert(text)
    toc_html = md.toc

    title = md_file.stem.replace("_", " ")
    for line in text.splitlines():
        if line.startswith("# "):
            title = line[2:].strip()
            break

    guide_key = normalize_guide_name(md_file.name)

    html_template = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{{TITLE}}</title>
  <meta name="description" content="Study {{TITLE}}, a step-by-step chapter of the Zero-to-Research Machine Learning Curriculum by Zeeshan Hayat.">
  <meta name="keywords" content="{{TITLE}}, Machine Learning, Zeeshan Hayat, Zero-to-Research, autograd, linear regression, GLM, AI">
  <meta name="author" content="Zeeshan Hayat">
  <meta name="robots" content="index, follow">
  
  <!-- Open Graph / Facebook -->
  <meta property="og:type" content="article">
  <meta property="og:title" content="{{TITLE}} | Zero-to-Research ML Curriculum">
  <meta property="og:description" content="Study {{TITLE}}, a step-by-step chapter of the Zero-to-Research Machine Learning Curriculum by Zeeshan Hayat.">
  <meta property="og:image" content="https://zeehayat.com/og-preview.png">

  <!-- Twitter -->
  <meta property="twitter:card" content="summary_large_image">
  <meta property="twitter:title" content="{{TITLE}} | Zero-to-Research ML Curriculum">
  <meta property="twitter:description" content="Study {{TITLE}}, a step-by-step chapter of the Zero-to-Research Machine Learning Curriculum by Zeeshan Hayat.">
  <meta property="twitter:image" content="https://zeehayat.com/og-preview.png">

  <style>
    :root {
      --ink: #1c232b;
      --muted: #5b6673;
      --line: #d9dee5;
      --paper: #ffffff;
      --soft: #f5f7fa;
      --accent: #0f766e;
      --accent2: #8b3a3a;
      --code: #111827;
      --codebg: #f0f3f6;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: system-ui, -apple-system, Segoe UI, Roboto, Arial, sans-serif;
      color: var(--ink);
      background: var(--paper);
      line-height: 1.62;
    }
    header {
      padding: 42px max(24px, calc((100vw - 1120px) / 2)) 28px;
      border-bottom: 1px solid var(--line);
      background: linear-gradient(180deg, #f7fbfb 0%, #fff 100%);
    }
    main {
      max-width: 1120px;
      margin: 0 auto;
      padding: 28px 24px 80px;
      display: grid;
      grid-template-columns: 260px minmax(0, 1fr);
      gap: 34px;
    }
    nav {
      position: sticky;
      top: 0;
      align-self: start;
      max-height: 100vh;
      overflow: auto;
      padding: 18px 0;
      border-right: 1px solid var(--line);
    }
    nav a, nav .toc a {
      display: block;
      color: var(--ink);
      text-decoration: none;
      padding: 5px 16px 5px 0;
      font-size: 14px;
    }
    nav a:hover, nav .toc a:hover { color: var(--accent); }
    nav .toc ul {
      list-style: none;
      padding-left: 0;
      margin: 0;
    }
    nav .toc li {
      margin: 0;
      padding: 0;
    }
    nav .toc ul ul {
      padding-left: 12px;
    }
    h1 {
      font-size: clamp(32px, 4vw, 54px);
      line-height: 1.05;
      margin: 0 0 12px;
      letter-spacing: 0;
    }
    h2 {
      margin: 44px 0 12px;
      padding-top: 10px;
      border-top: 2px solid var(--line);
      font-size: 30px;
      line-height: 1.2;
      letter-spacing: 0;
    }
    h3 {
      margin: 28px 0 8px;
      font-size: 21px;
      letter-spacing: 0;
    }
    h4 { margin: 20px 0 6px; font-size: 17px; }
    p, li { max-width: 78ch; }
    .lede { max-width: 86ch; color: var(--muted); font-size: 18px; }
    .box {
      background: var(--soft);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 16px 18px;
      margin: 16px 0;
    }
    .warn { border-left: 5px solid var(--accent2); }
    .key { border-left: 5px solid var(--accent); }
    .review-note {
      border-left: 5px solid #5b5f97;
      background: #f7f7fc;
    }
    code, pre { font-family: ui-monospace, SFMono-Regular, Consolas, "Liberation Mono", monospace; }
    code { background: var(--codebg); padding: 1px 4px; border-radius: 4px; }
    pre {
      overflow: auto;
      background: var(--code);
      color: #f8fafc;
      padding: 16px;
      border-radius: 8px;
      line-height: 1.45;
      font-size: 13px;
    }
    pre code { background: transparent; padding: 0; color: inherit; }
    table {
      border-collapse: collapse;
      width: 100%;
      margin: 16px 0 22px;
      font-size: 14px;
    }
    th, td { border: 1px solid var(--line); padding: 9px 10px; vertical-align: top; }
    th { background: var(--soft); text-align: left; }
    .formula {
      font-family: ui-monospace, SFMono-Regular, Consolas, monospace;
      background: #fbfbfd;
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 12px 14px;
      overflow: auto;
      white-space: pre-wrap;
    }
    .exercise {
      border-left: 4px solid var(--accent);
      padding: 9px 14px;
      background: #f6fbfa;
      margin: 12px 0 20px;
    }
    .term {
      border-top: 1px solid var(--line);
      padding-top: 14px;
      margin-top: 18px;
    }
    .expand-toolbar {
      display: flex;
      gap: 8px;
      flex-wrap: wrap;
      margin: 14px 0 18px;
    }
    .expand-toolbar button {
      border: 1px solid #94a3b8;
      border-radius: 6px;
      padding: 8px 11px;
      background: #fff;
      color: var(--ink);
      font: inherit;
      cursor: pointer;
    }
    details.deep-dive {
      max-width: 850px;
      margin: 12px 0;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: #fff;
      overflow: hidden;
    }
    details.deep-dive > summary {
      cursor: pointer;
      padding: 12px 14px;
      background: #f8fafc;
      font-weight: 700;
      list-style-position: inside;
    }
    details.deep-dive[open] > summary {
      border-bottom: 1px solid var(--line);
      background: #eef7f6;
      color: #0f5f59;
    }
    .deep-dive-body {
      padding: 14px 16px 16px;
    }
    .deep-dive-body h4:first-child { margin-top: 0; }
    .mental-model {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
      gap: 12px;
      margin: 14px 0;
      max-width: 850px;
    }
    .mental-model > div {
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 12px;
      background: #fbfcfd;
    }
    .mental-model h4 { margin-top: 0; }
    .notes-panel {
      background: #fbfcfd;
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 18px;
      margin: 20px 0 28px;
      max-width: 850px;
    }
    .notes-grid {
      display: grid;
      grid-template-columns: minmax(0, 1fr) minmax(0, 1fr);
      gap: 14px;
    }
    .notes-panel label {
      display: block;
      font-weight: 650;
      margin-bottom: 5px;
    }
    .notes-panel input,
    .notes-panel select,
    .notes-panel textarea {
      width: 100%;
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 9px 10px;
      font: inherit;
      background: #fff;
      color: var(--ink);
    }
    .notes-panel textarea {
      min-height: 150px;
      resize: vertical;
    }
    .notes-actions {
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      margin-top: 12px;
    }
    .notes-actions button {
      border: 1px solid #94a3b8;
      border-radius: 6px;
      padding: 8px 11px;
      background: #fff;
      color: var(--ink);
      font: inherit;
      cursor: pointer;
    }
    .notes-actions button.primary {
      border-color: var(--accent);
      background: var(--accent);
      color: #fff;
    }
    .notes-status {
      color: var(--muted);
      font-size: 14px;
      margin-top: 10px;
    }
    .notes-list {
      display: grid;
      gap: 10px;
      margin-top: 16px;
    }
    .topic-note-button {
      margin-left: 8px;
      border: 1px solid var(--accent);
      border-radius: 999px;
      background: #fff;
      color: var(--accent);
      font: 12px system-ui, -apple-system, Segoe UI, Roboto, Arial, sans-serif;
      padding: 3px 8px;
      vertical-align: middle;
      cursor: pointer;
    }
    .floating-note-button {
      position: fixed;
      right: 22px;
      bottom: 22px;
      z-index: 20;
      border: 1px solid var(--accent);
      border-radius: 999px;
      background: var(--accent);
      color: #fff;
      font: 700 15px system-ui, -apple-system, Segoe UI, Roboto, Arial, sans-serif;
      padding: 12px 16px;
      box-shadow: 0 10px 30px rgba(15, 23, 42, 0.22);
      cursor: pointer;
    }
    .floating-note-panel {
      position: fixed;
      right: 22px;
      bottom: 78px;
      z-index: 21;
      width: min(430px, calc(100vw - 44px));
      max-height: calc(100vh - 110px);
      overflow: auto;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: #fff;
      box-shadow: 0 18px 50px rgba(15, 23, 42, 0.24);
      padding: 16px;
    }
    .floating-note-panel[hidden] {
      display: none;
    }
    .floating-note-panel header {
      padding: 0 0 10px;
      border: 0;
      background: transparent;
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 12px;
    }
    .floating-note-panel h3 {
      margin: 0;
      font-size: 18px;
    }
    .floating-note-panel label {
      display: block;
      margin: 10px 0 5px;
      font-weight: 650;
    }
    .floating-note-panel input,
    .floating-note-panel select,
    .floating-note-panel textarea {
      width: 100%;
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 9px 10px;
      font: inherit;
    }
    .floating-note-panel textarea {
      min-height: 170px;
      resize: vertical;
    }
    .floating-topic-context {
      color: var(--muted);
      font-size: 13px;
      margin: 0 0 6px;
    }
    .close-note-panel {
      border: 1px solid var(--line);
      border-radius: 6px;
      background: #fff;
      cursor: pointer;
      font: inherit;
      padding: 4px 8px;
    }
    .note-card {
      border: 1px solid var(--line);
      border-radius: 8px;
      background: #fff;
      padding: 12px;
    }
    .note-card header {
      padding: 0;
      border: 0;
      background: transparent;
      display: flex;
      justify-content: space-between;
      gap: 12px;
      align-items: baseline;
    }
    .note-card h4 {
      margin: 0;
      font-size: 16px;
    }
    .note-card .meta {
      color: var(--muted);
      font-size: 12px;
      white-space: nowrap;
    }
    .note-card p {
      margin: 8px 0 0;
      white-space: pre-wrap;
    }
    .home-link {
      display: inline-block;
      margin-bottom: 20px;
      color: var(--accent);
      text-decoration: none;
      font-weight: bold;
    }
    .home-link:hover {
      text-decoration: underline;
    }
    @media (max-width: 720px) {
      .notes-grid { grid-template-columns: 1fr; }
    }
    @media (max-width: 880px) {
      main { display: block; }
      nav { position: static; max-height: none; border-right: 0; border-bottom: 1px solid var(--line); }
    }
  </style>
</head>
<body>
<header>
  <a href="/" class="home-link">← Hub Dashboard</a>
  <h1>{{TITLE}}</h1>
</header>
<main>
  <nav aria-label="Table of contents">
    <a href="#notes">Notes</a>
    {{TOC}}
  </nav>
  <article>
    <section id="notes">
      <h2>Offline Notes</h2>
      <p>Use this panel when the guide is opened through the local notes server. Notes are stored in <code>regression_notes.sqlite3</code> in the same folder as this HTML file.</p>
      <div class="notes-panel" data-notes-app>
        <div class="notes-grid">
          <div>
            <label for="note-section">Section</label>
            <select id="note-section">
              <option>General</option>
            </select>
          </div>
          <div>
            <label for="note-title">Title</label>
            <input id="note-title" type="text" placeholder="Example: Key concept definition">
          </div>
        </div>
        <label for="note-body" style="margin-top: 12px;">Note</label>
        <textarea id="note-body" placeholder="Write your note, question, derivation, or teaching prompt here."></textarea>
        <div class="notes-actions">
          <button class="primary" type="button" id="save-note">Save note</button>
          <button type="button" id="refresh-notes">Refresh</button>
          <button type="button" id="export-notes">Export JSON</button>
          <button type="button" id="clear-note-form">Clear form</button>
        </div>
        <div class="notes-status" id="notes-status">Notes save to SQLite when the local server is running.</div>
        <div class="notes-list" id="notes-list" aria-live="polite"></div>
      </div>
    </section>

    {{BODY}}
  </article>
</main>

<button class="floating-note-button" type="button" id="open-note-panel">New note</button>
<aside class="floating-note-panel" id="floating-note-panel" hidden aria-label="Create topic note">
  <header>
    <h3>Topic Note</h3>
    <button class="close-note-panel" type="button" id="close-note-panel">Close</button>
  </header>
  <p class="floating-topic-context" id="floating-topic-context">Topic: General</p>
  <label for="floating-note-section">Section</label>
  <select id="floating-note-section">
    <option>General</option>
  </select>
  <label for="floating-note-title">Title</label>
  <input id="floating-note-title" type="text" placeholder="My understanding of this topic">
  <label for="floating-note-body">Note</label>
  <textarea id="floating-note-body" placeholder="Explain the topic in your own words, add questions, derivations, or examples."></textarea>
  <div class="notes-actions">
    <button class="primary" type="button" id="floating-save-note">Save note</button>
    <button type="button" id="floating-clear-note">Clear</button>
  </div>
  <div class="notes-status" id="floating-notes-status">Notes save to SQLite when the local server is running.</div>
</aside>

<script>
(() => {
  const guideKey = "{{GUIDE_KEY}}";
  
  const sectionEl = document.getElementById("note-section");
  const titleEl = document.getElementById("note-title");
  const bodyEl = document.getElementById("note-body");
  const saveBtn = document.getElementById("save-note");
  const refreshBtn = document.getElementById("refresh-notes");
  const clearBtn = document.getElementById("clear-note-form");
  const exportBtn = document.getElementById("export-notes");
  const listEl = document.getElementById("notes-list");
  const statusEl = document.getElementById("notes-status");

  const openPanelBtn = document.getElementById("open-note-panel");
  const closePanelBtn = document.getElementById("close-note-panel");
  const floatingPanel = document.getElementById("floating-note-panel");
  const floatingContextEl = document.getElementById("floating-topic-context");
  const floatingSectionEl = document.getElementById("floating-note-section");
  const floatingTitleEl = document.getElementById("floating-note-title");
  const floatingBodyEl = document.getElementById("floating-note-body");
  const floatingSaveBtn = document.getElementById("floating-save-note");
  const floatingClearBtn = document.getElementById("floating-clear-note");
  const floatingStatusEl = document.getElementById("floating-notes-status");

  let editingId = null;
  let floatingEditingId = null;
  let currentTopic = { section: "General", topic_anchor: "", topic_title: "General" };
  const topicButtons = new Map();

  function setStatus(message) {
    statusEl.textContent = message;
    if (floatingStatusEl) floatingStatusEl.textContent = message;
  }

  function escapeText(value) {
    return String(value ?? "");
  }

  function slugify(value) {
    return String(value)
      .toLowerCase()
      .replace(/[^a-z0-9]+/g, "-")
      .replace(/^-+|-+$/g, "")
      .slice(0, 90) || "topic";
  }

  function cleanHeadingText(heading) {
    return Array.from(heading.childNodes)
      .filter((node) => node.nodeType === Node.TEXT_NODE)
      .map((node) => node.textContent)
      .join(" ")
      .replace(/\\s+/g, " ")
      .trim();
  }

  function sectionNameFor(heading) {
    if (heading.tagName === "H2") {
      return cleanHeadingText(heading).replace(/^\\d+\\.\\s*/, "") || "General";
    }
    let prev = heading.previousElementSibling;
    while (prev) {
      if (prev.tagName === "H2") {
        return cleanHeadingText(prev).replace(/^\\d+\\.\\s*/, "") || "General";
      }
      prev = prev.previousElementSibling;
    }
    return "General";
  }

  function populateSectionOptions(sections) {
    const selectElements = [sectionEl, floatingSectionEl];
    selectElements.forEach(select => {
      if (!select) return;
      select.innerHTML = '<option>General</option>';
      sections.forEach(sec => {
        if (sec && sec !== "General") {
          const opt = document.createElement("option");
          opt.textContent = sec;
          select.appendChild(opt);
        }
      });
    });
  }

  function ensureTopicControls() {
    const headings = document.querySelectorAll("article h2, article h3");
    const used = new Set();
    const sections = new Set();

    headings.forEach((heading) => {
      const topicTitle = cleanHeadingText(heading);
      if (!topicTitle) return;
      if (heading.closest("#notes")) return;
      
      const secName = sectionNameFor(heading);
      sections.add(secName);

      if (!heading.id) {
        let base = slugify(topicTitle);
        let id = base;
        let i = 2;
        while (document.getElementById(id) || used.has(id)) {
          id = `${base}-${i}`;
          i += 1;
        }
        heading.id = id;
        used.add(id);
      }

      const button = document.createElement("button");
      button.type = "button";
      button.className = "topic-note-button";
      button.textContent = "Add note";
      button.title = `Add a note for: ${topicTitle}`;
      button.addEventListener("click", () => {
        openFloatingPanel({
          section: secName,
          topic_anchor: heading.id,
          topic_title: topicTitle
        });
      });
      heading.appendChild(button);
      topicButtons.set(heading.id, button);
    });
    
    populateSectionOptions(Array.from(sections));
  }

  async function api(path, options = {}) {
    const response = await fetch(path, {
      headers: { "Content-Type": "application/json", ...(options.headers || {}) },
      ...options
    });
    if (!response.ok) {
      const text = await response.text();
      throw new Error(text || `Request failed: ${response.status}`);
    }
    return response.json();
  }

  function clearForm() {
    editingId = null;
    titleEl.value = "";
    bodyEl.value = "";
    sectionEl.value = "General";
    saveBtn.textContent = "Save note";
  }

  function clearFloatingForm(keepTopic = true) {
    floatingEditingId = null;
    floatingTitleEl.value = "";
    floatingBodyEl.value = "";
    floatingSaveBtn.textContent = "Save note";
    if (!keepTopic) {
      currentTopic = { section: "General", topic_anchor: "", topic_title: "General" };
      floatingSectionEl.value = "General";
      floatingContextEl.textContent = "Topic: General";
    }
  }

  function openFloatingPanel(topic, note = null) {
    currentTopic = {
      section: topic.section || "General",
      topic_anchor: topic.topic_anchor || "",
      topic_title: topic.topic_title || "General"
    };
    floatingPanel.hidden = false;
    floatingSectionEl.value = currentTopic.section;
    floatingContextEl.textContent = `Topic: ${currentTopic.topic_title}`;

    if (note) {
      floatingEditingId = note.id;
      floatingTitleEl.value = note.title || "";
      floatingBodyEl.value = note.body || "";
      floatingSaveBtn.textContent = "Update note";
    } else {
      clearFloatingForm(true);
    }
    floatingTitleEl.focus();
  }

  function updateTopicButtonCounts(notes) {
    const counts = new Map();
    for (const note of notes) {
      if (!note.topic_anchor) continue;
      counts.set(note.topic_anchor, (counts.get(note.topic_anchor) || 0) + 1);
    }
    for (const [anchor, button] of topicButtons.entries()) {
      const count = counts.get(anchor) || 0;
      button.textContent = count ? `Add note (${count})` : "Add note";
    }
  }

  function renderNotes(notes) {
    listEl.innerHTML = "";
    updateTopicButtonCounts(notes);
    if (!notes.length) {
      const empty = document.createElement("p");
      empty.className = "notes-status";
      empty.textContent = "No notes saved yet.";
      listEl.appendChild(empty);
      return;
    }

    for (const note of notes) {
      const card = document.createElement("article");
      card.className = "note-card";

      const header = document.createElement("header");
      const title = document.createElement("h4");
      title.textContent = note.title || "(Untitled note)";
      const meta = document.createElement("span");
      meta.className = "meta";
      const topic = note.topic_title ? ` > ${note.topic_title}` : "";
      meta.textContent = `${note.section}${topic} | ${note.updated_at}`;
      header.append(title, meta);

      const body = document.createElement("p");
      body.textContent = escapeText(note.body);

      const actions = document.createElement("div");
      actions.className = "notes-actions";

      const edit = document.createElement("button");
      edit.type = "button";
      edit.textContent = "Edit";
      edit.addEventListener("click", () => {
        openFloatingPanel({
          section: note.section || "General",
          topic_anchor: note.topic_anchor || "",
          topic_title: note.topic_title || "General"
        }, note);
      });

      if (note.topic_anchor) {
        const jump = document.createElement("button");
        jump.type = "button";
        jump.textContent = "Go to topic";
        jump.addEventListener("click", () => {
          document.getElementById(note.topic_anchor)?.scrollIntoView({ behavior: "smooth", block: "start" });
        });
        actions.appendChild(jump);
      }

      const remove = document.createElement("button");
      remove.type = "button";
      remove.textContent = "Delete";
      remove.addEventListener("click", async () => {
        if (!confirm("Delete this note from the SQLite database?")) return;
        try {
          await api(`/api/notes/${note.id}`, { method: "DELETE" });
          setStatus("Note deleted.");
          await loadNotes();
        } catch (err) {
          setStatus(`Delete failed: ${err.message}`);
        }
      });

      actions.append(edit, remove);
      card.append(header, body, actions);
      listEl.appendChild(card);
    }
  }

  async function loadNotes() {
    try {
      const data = await api(`/api/notes?guide=${guideKey}`);
      renderNotes(data.notes || []);
      setStatus(`SQLite notes ready. Database: ${data.database}`);
    } catch (err) {
      renderNotes([]);
      setStatus("Notes are offline.");
    }
  }

  saveBtn.addEventListener("click", async () => {
    const title = titleEl.value.trim();
    const body = bodyEl.value.trim();
    if (!title && !body) {
      setStatus("Write a title or note body before saving.");
      return;
    }

    try {
      await api("/api/notes", {
        method: "POST",
        body: JSON.stringify({
          id: editingId,
          guide: guideKey,
          section: sectionEl.value,
          topic_anchor: "",
          topic_title: "",
          title,
          body
        })
      });
      clearForm();
      setStatus("Note saved to SQLite.");
      await loadNotes();
    } catch (err) {
      setStatus(`Save failed: ${err.message}`);
    }
  });

  refreshBtn.addEventListener("click", loadNotes);
  clearBtn.addEventListener("click", clearForm);
  exportBtn.addEventListener("click", () => {
    window.location.href = `/api/notes/export?guide=${guideKey}`;
  });
  openPanelBtn.addEventListener("click", () => openFloatingPanel({
    section: "General",
    topic_anchor: "",
    topic_title: "General"
  }));
  closePanelBtn.addEventListener("click", () => {
    floatingPanel.hidden = true;
  });
  floatingClearBtn.addEventListener("click", () => clearFloatingForm(true));
  floatingSaveBtn.addEventListener("click", async () => {
    const title = floatingTitleEl.value.trim();
    const body = floatingBodyEl.value.trim();
    if (!title && !body) {
      setStatus("Write a title or note body before saving.");
      return;
    }

    try {
      await api("/api/notes", {
        method: "POST",
        body: JSON.stringify({
          id: floatingEditingId,
          guide: guideKey,
          section: floatingSectionEl.value || currentTopic.section,
          topic_anchor: currentTopic.topic_anchor,
          topic_title: currentTopic.topic_title,
          title,
          body
        })
      });
      clearFloatingForm(true);
      floatingPanel.hidden = true;
      setStatus("Topic note saved to SQLite.");
      await loadNotes();
    } catch (err) {
      setStatus(`Save failed: ${err.message}`);
    }
  });

  ensureTopicControls();
  loadNotes();

  // --- Reading Progress & Resume State Logic ---
  const resumeStyles = `
    .resume-modal-overlay {
      position: fixed;
      top: 0;
      left: 0;
      width: 100vw;
      height: 100vh;
      background: rgba(11, 19, 41, 0.6);
      backdrop-filter: blur(8px);
      -webkit-backdrop-filter: blur(8px);
      display: flex;
      justify-content: center;
      align-items: center;
      z-index: 10000;
      opacity: 0;
      transition: opacity 0.3s ease;
    }
    .resume-modal-overlay.active {
      opacity: 1;
    }
    .resume-modal-container {
      border-radius: 16px;
      padding: 32px;
      width: min(500px, 90vw);
      transform: scale(0.9);
      transition: transform 0.3s cubic-bezier(0.34, 1.56, 0.64, 1);
      font-family: system-ui, -apple-system, sans-serif;
      text-align: center;
    }
    .resume-modal-overlay.active .resume-modal-container {
      transform: scale(1);
    }
    .resume-modal-overlay.theme-hub {
      --bg: rgba(28, 37, 65, 0.95);
      --border: 1px solid rgba(91, 192, 190, 0.3);
      --text: #ffffff;
      --text-muted: #a5b4fc;
      --shadow: 0 20px 40px rgba(0, 0, 0, 0.5), 0 0 25px rgba(91, 192, 190, 0.2);
      --primary-bg: #5bc0be;
      --primary-text: #0b1329;
      --primary-hover: #75d4d2;
      --secondary-bg: rgba(255, 255, 255, 0.05);
      --secondary-border: 1px solid rgba(255, 255, 255, 0.15);
      --secondary-text: #ffffff;
      --secondary-hover: rgba(255, 255, 255, 0.12);
    }
    .resume-modal-overlay.theme-guide {
      --bg: rgba(255, 255, 255, 0.98);
      --border: 1px solid rgba(15, 118, 110, 0.25);
      --text: #1c232b;
      --text-muted: #5b6673;
      --shadow: 0 20px 40px rgba(15, 23, 42, 0.15), 0 0 25px rgba(15, 118, 110, 0.1);
      --primary-bg: #0f766e;
      --primary-text: #ffffff;
      --primary-hover: #115e59;
      --secondary-bg: #f8fafc;
      --secondary-border: 1px solid #cbd5e1;
      --secondary-text: #334155;
      --secondary-hover: #e2e8f0;
    }
    .resume-modal-container {
      background: var(--bg);
      border: var(--border);
      color: var(--text);
      box-shadow: var(--shadow);
    }
    .resume-modal-title {
      font-size: 24px;
      font-weight: 800;
      margin: 0 0 16px;
      letter-spacing: -0.5px;
    }
    .resume-modal-body {
      font-size: 15px;
      line-height: 1.6;
      margin: 0 0 28px;
      color: var(--text-muted);
    }
    .resume-location-badge {
      display: inline-block;
      margin-top: 12px;
      padding: 6px 12px;
      border-radius: 8px;
      background: rgba(15, 118, 110, 0.1);
      color: #0f766e;
      font-weight: 600;
      font-size: 13px;
    }
    .theme-hub .resume-location-badge {
      background: rgba(91, 192, 190, 0.1);
      color: #5bc0be;
    }
    .resume-modal-actions {
      display: flex;
      gap: 12px;
      justify-content: center;
    }
    .resume-btn {
      padding: 12px 24px;
      font-size: 14px;
      font-weight: 700;
      border-radius: 8px;
      cursor: pointer;
      transition: all 0.2s ease;
      font-family: inherit;
    }
    .resume-btn-primary {
      background: var(--primary-bg);
      color: var(--primary-text);
      border: none;
    }
    .resume-btn-primary:hover {
      background: var(--primary-hover);
      transform: translateY(-2px);
      box-shadow: 0 4px 12px rgba(0,0,0,0.1);
    }
    .resume-btn-primary:active {
      transform: translateY(0);
    }
    .resume-btn-secondary {
      background: var(--secondary-bg);
      border: var(--secondary-border);
      color: var(--secondary-text);
    }
    .resume-btn-secondary:hover {
      background: var(--secondary-hover);
      color: var(--secondary-text);
    }
  `;

  function getGuideKeyFromPath() {
    const path = window.location.pathname;
    const filename = path.substring(path.lastIndexOf('/') + 1);
    if (!filename || filename === "index.html" || filename === "") {
      return "hub";
    }
    let name = filename.replace(/\\.html$/, "").replace(/\\.md$/, "");
    name = name.toLowerCase().replace(/_/g, "").replace(/-/g, "");
    if (name.endsWith("guide") && name !== "regressionguide") {
      name = name.substring(0, name.length - 5);
    }
    if (name === "regressionguide") {
      name = "regression";
    }
    return name || "general";
  }

  function getGuideDisplayName(key) {
    const names = {
      "regression": "Linear & Logistic Regression Guide",
      "generalizedlinearmodelsmathematicalandpython": "Generalized Linear Models Guide",
      "nexttopicagentbrief": "Next Topic Agent Brief",
      "notesreadme": "Notes Readme Guide"
    };
    return names[key] || key.charAt(0).toUpperCase() + key.slice(1);
  }

  function getActiveHeading() {
    const headings = Array.from(document.querySelectorAll("h2, h3, h4"));
    let active = null;
    for (const h of headings) {
      if (!h.id || h.closest("#notes")) continue;
      const rect = h.getBoundingClientRect();
      if (rect.top <= 150) {
        active = h;
      } else {
        break;
      }
    }
    return active;
  }

  let lastSavedState = null;
  let saveTimeout = null;

  function trackReadingProgress() {
    const pathname = window.location.pathname;
    if (pathname === "/" || pathname === "/index.html" || pathname === "") return;

    const activeHeading = getActiveHeading();
    const anchor = activeHeading ? "#" + activeHeading.id : "";
    let headingTitle = activeHeading ? activeHeading.textContent : "";
    if (headingTitle && headingTitle.includes("Add note")) {
      headingTitle = headingTitle.replace("Add note", "").trim();
    }

    const scrollY = window.scrollY;
    const docHeight = document.documentElement.scrollHeight - window.innerHeight;
    const scrollPercent = docHeight > 0 ? (scrollY / docHeight) * 100 : 0;
    const guide = getGuideKeyFromPath();

    const state = {
      guide,
      pathname,
      anchor,
      title: headingTitle || document.title,
      scroll_y: scrollY,
      scroll_percent: parseFloat(scrollPercent.toFixed(2))
    };

    if (
      !lastSavedState ||
      lastSavedState.pathname !== state.pathname ||
      lastSavedState.anchor !== state.anchor ||
      Math.abs(lastSavedState.scroll_y - state.scroll_y) > 100
    ) {
      lastSavedState = state;
      localStorage.setItem("last_reading_state", JSON.stringify(state));

      clearTimeout(saveTimeout);
      saveTimeout = setTimeout(async () => {
        try {
          await fetch("/api/reading-state", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(state)
          });
        } catch (err) {
          console.error("Failed to save reading state:", err);
        }
      }, 1500);
    }
  }

  function restoreScrollPosition() {
    const restoreY = localStorage.getItem("restore_scroll");
    if (restoreY !== null) {
      localStorage.removeItem("restore_scroll");
      localStorage.removeItem("restore_scroll_percent");
      const targetY = parseInt(restoreY, 10);
      if (!isNaN(targetY)) {
        window.scrollTo(0, targetY);
        setTimeout(() => window.scrollTo(0, targetY), 300);
        setTimeout(() => window.scrollTo(0, targetY), 800);
      }
    }
  }

  async function initResumePrompt() {
    if (sessionStorage.getItem("resume_prompt_dismissed")) return;

    const pathname = window.location.pathname;
    const isHub = pathname === "/" || pathname === "/index.html" || pathname === "";
    const currentGuide = getGuideKeyFromPath();

    let state = null;
    try {
      const url = isHub ? "/api/reading-state" : "/api/reading-state?guide=" + currentGuide;
      const res = await fetch(url);
      if (res.ok) {
        const data = await res.json();
        state = data.state;
      }
    } catch (err) {
      console.warn("Failed to fetch reading state:", err);
    }

    if (!state) {
      try {
        const local = localStorage.getItem("last_reading_state");
        if (local) {
          const parsed = JSON.parse(local);
          if (isHub || parsed.guide === currentGuide) {
            state = parsed;
          }
        }
      } catch (e) {}
    }

    if (!state) return;

    if (!isHub) {
      if (window.scrollY > 200) return;
      if (state.scroll_y < 200) return;
      if (localStorage.getItem("restore_scroll")) return;
    }

    const styleEl = document.createElement("style");
    styleEl.textContent = resumeStyles;
    document.head.appendChild(styleEl);

    const overlay = document.createElement("div");
    overlay.className = "resume-modal-overlay theme-" + (isHub ? "hub" : "guide");

    const container = document.createElement("div");
    container.className = "resume-modal-container";

    const titleEl = document.createElement("h3");
    titleEl.className = "resume-modal-title";
    titleEl.textContent = "Resume Reading?";

    const bodyEl = document.createElement("div");
    bodyEl.className = "resume-modal-body";
    
    const guideName = getGuideDisplayName(state.guide);
    const locationText = state.title ? "<br><span class=\"resume-location-badge\">" + state.title + "</span>" : "";
    bodyEl.innerHTML = "Welcome back! Would you like to resume reading <strong>" + guideName + "</strong>" + locationText + ", or start from the beginning?";

    const actions = document.createElement("div");
    actions.className = "resume-modal-actions";

    const btnResume = document.createElement("button");
    btnResume.className = "resume-btn resume-btn-primary";
    btnResume.textContent = "Resume Reading";
    btnResume.addEventListener("click", () => {
      overlay.classList.remove("active");
      setTimeout(() => overlay.remove(), 300);

      if (isHub) {
        localStorage.setItem("restore_scroll", state.scroll_y);
        localStorage.setItem("restore_scroll_percent", state.scroll_percent);
        window.location.href = state.pathname + (state.anchor || "");
      } else {
        window.scrollTo({ top: state.scroll_y, behavior: "smooth" });
      }
    });

    const btnStart = document.createElement("button");
    btnStart.className = "resume-btn resume-btn-secondary";
    btnStart.textContent = "Start from Beginning";
    btnStart.addEventListener("click", () => {
      sessionStorage.setItem("resume_prompt_dismissed", "true");
      overlay.classList.remove("active");
      setTimeout(() => overlay.remove(), 300);
    });

    actions.append(btnResume, btnStart);
    container.append(titleEl, bodyEl, actions);
    overlay.appendChild(container);
    document.body.appendChild(overlay);

    setTimeout(() => overlay.classList.add("active"), 50);
  }

  window.addEventListener("scroll", trackReadingProgress, { passive: true });
  window.addEventListener("hashchange", trackReadingProgress);
  window.addEventListener("load", () => {
    setTimeout(restoreScrollPosition, 100);
    setTimeout(initResumePrompt, 500);
  });
})();
</script>
</body>
</html>
"""

    return (
        html_template.replace("{{TITLE}}", title)
        .replace("{{TOC}}", toc_html)
        .replace("{{BODY}}", html_body)
        .replace("{{GUIDE_KEY}}", guide_key)
    )


init_db()
migrate_sqlite_to_postgres()
app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "zeehayat_secret_key_123984")

@app.before_request
def enforce_login():
    # If logged in, proceed
    if session.get("user_email"):
        return
        
    endpoint = request.endpoint
    if not endpoint:
        return
        
    # Allow access to login, callback, status, logout, robots.txt, and sitemap.xml endpoints
    allowed_endpoints = ["login_page", "login_email", "google_callback", "auth_status", "logout", "robots_txt", "sitemap_xml"]
    if endpoint in allowed_endpoints:
        return
        
    # Block API access to notes and progress for unauthenticated users
    blocked_endpoints = ["list_notes", "save_note", "delete_note", "export_notes", "get_reading_state", "save_reading_state"]
    if endpoint in blocked_endpoints:
        return jsonify({"error": "Unauthorized", "message": "Please sign in first"}), 401
        
    # Block access to book files (HTML guides)
    if endpoint == "serve_static":
        filename = request.view_args.get("filename", "")
        if filename.lower().endswith(".html"):
            return redirect("/login")

def serve_hub() -> str:
    counts = get_note_counts()
    cards_html = []
    
    # OLS static Regression Guide
    reg_count = counts.get("regression", 0)
    reg_desc = GUIDE_DESCRIPTIONS.get("regression", "")
    cards_html.append(f"""
    <a class="card" href="/regression_guide.html">
      <div>
        <h2>Zero-to-Research Machine Learning Textbook (Complete)</h2>
        <p>{reg_desc}</p>
      </div>
      <div class="card-meta">
        <span class="notes-badge">{reg_count} notes</span>
        <span class="open-btn">Open Guide</span>
      </div>
    </a>
    """)

    # Other markdown files
    for f in sorted(ROOT.glob("*.md")):
        norm_name = normalize_guide_name(f.name)
        title = get_guide_title(f.name)
        desc = GUIDE_DESCRIPTIONS.get(norm_name, "Learn about this topic in the zero-to-research ML curriculum.")
        count = counts.get(norm_name, 0)
        href = f"{f.stem}.html"
        
        cards_html.append(f"""
        <a class="card" href="/{href}">
          <div>
            <h2>{title}</h2>
            <p>{desc}</p>
          </div>
          <div class="card-meta">
            <span class="notes-badge">{count} notes</span>
            <span class="open-btn">Open Guide</span>
          </div>
        </a>
        """)

    user_email = session.get("user_email")
    if user_email:
        nav_html = f"""
        <a href="#">Home</a>
        <a href="#about">About</a>
        <a href="#guides">Guides</a>
        <a href="#contact">Contact</a>
        <span class="user-badge" style="background: rgba(6,182,212,0.1); color: var(--accent); padding: 4px 10px; border-radius: 6px; font-size: 13px; font-weight: 600; max-width: 180px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap;">{user_email}</span>
        <a href="/logout" style="color: #ef4444; font-weight: 600;">Logout</a>
        """
    else:
        nav_html = """
        <a href="#">Home</a>
        <a href="#about">About</a>
        <a href="#guides">Guides</a>
        <a href="#contact">Contact</a>
        <a href="/login" style="background: var(--accent-gradient); color: #030712; padding: 4px 10px; border-radius: 6px; font-weight: 600; text-decoration: none; display: inline-block;">Sign In</a>
        """

    template = HUB_TEMPLATE.replace("<!-- Cards will be populated dynamically -->", "\n".join(cards_html))
    template = template.replace("<!-- SESSION_LINKS -->", nav_html)
    return template

@app.route("/")
@app.route("/index.html")
def serve_hub_route():
    response = make_response(serve_hub())
    response.headers["Cache-Control"] = "no-store"
    return response

@app.route("/api/notes", methods=["GET"])
def list_notes():
    guide = request.args.get("guide")
    if not guide:
        referer = request.headers.get("Referer", "")
        if referer:
            ref_path = urlparse(referer).path
            guide = Path(ref_path).stem or "regression"
        else:
            guide = "regression"
    
    guide = normalize_guide_name(guide)
    user_email = get_current_user_email()
    rows, _ = db_execute(
        """
        SELECT id, guide, section, title, body, created_at, updated_at
        , topic_anchor, topic_title
        FROM notes
        WHERE guide = ? AND user_email = ?
        ORDER BY updated_at DESC, id DESC
        """,
        (guide, user_email),
    )
    
    response = jsonify({
        "database": "postgresql" if os.environ.get("DB_HOST") else str(DB_FILE),
        "notes": rows,
    })
    response.headers["Cache-Control"] = "no-store"
    return response

@app.route("/api/notes", methods=["POST"])
def save_note():
    payload = request.get_json() or {}
    note_id = payload.get("id")
    section = str(payload.get("section") or "General").strip()[:120] or "General"
    topic_anchor = str(payload.get("topic_anchor") or "").strip()[:160]
    topic_title = str(payload.get("topic_title") or "").strip()[:240]
    title = str(payload.get("title") or "").strip()[:240]
    body = str(payload.get("body") or "").strip()
    user_email = get_current_user_email()
    
    if not title and not body:
        return "Title or body is required", 400

    guide = payload.get("guide")
    if not guide:
        referer = request.headers.get("Referer", "")
        if referer:
            ref_path = urlparse(referer).path
            guide = Path(ref_path).stem or "regression"
        else:
            guide = "regression"
            
    guide = normalize_guide_name(guide)
    now = utc_now()
    if note_id:
        # Ensure ownership
        rows, _ = db_execute("SELECT user_email FROM notes WHERE id = ?", (int(note_id),))
        if rows and rows[0]["user_email"] != user_email:
            return "Permission denied", 403
        db_execute(
            """
            UPDATE notes
            SET section = ?, topic_anchor = ?, topic_title = ?, title = ?, body = ?, updated_at = ?
            WHERE id = ? AND user_email = ?
            """,
            (section, topic_anchor, topic_title, title, body, now, int(note_id), user_email),
        )
        saved_id = int(note_id)
    else:
        _, saved_id = db_execute(
            """
            INSERT INTO notes(guide, section, topic_anchor, topic_title, title, body, user_email, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (guide, section, topic_anchor, topic_title, title, body, user_email, now, now),
        )
            
    return jsonify({"ok": True, "id": saved_id})

@app.route("/api/notes/<int:note_id>", methods=["DELETE"])
def delete_note(note_id):
    user_email = get_current_user_email()
    rows, _ = db_execute("SELECT user_email FROM notes WHERE id = ?", (note_id,))
    if rows and rows[0]["user_email"] != user_email:
        return "Permission denied", 403
    db_execute("DELETE FROM notes WHERE id = ? AND user_email = ?", (note_id, user_email))
    return jsonify({"ok": True})

@app.route("/api/notes/export", methods=["GET"])
def export_notes():
    guide = request.args.get("guide")
    if not guide:
        referer = request.headers.get("Referer", "")
        if referer:
            ref_path = urlparse(referer).path
            guide = Path(ref_path).stem or "regression"
        else:
            guide = "regression"
            
    guide = normalize_guide_name(guide)
    user_email = get_current_user_email()
    rows, _ = db_execute(
        """
        SELECT id, guide, section, title, body, created_at, updated_at
        , topic_anchor, topic_title
        FROM notes
        WHERE guide = ? AND user_email = ?
        ORDER BY section ASC, topic_title ASC, updated_at DESC, id DESC
        """,
        (guide, user_email),
    )
        
    data = json.dumps(rows, indent=2).encode("utf-8")
    response = make_response(data)
    response.headers["Content-Type"] = "application/json; charset=utf-8"
    response.headers["Content-Disposition"] = f'attachment; filename="{guide}_notes_export.json"'
    response.headers["Content-Length"] = str(len(data))
    return response

@app.route("/api/reading-state", methods=["GET"])
def get_reading_state():
    guide = request.args.get("guide")
    user_email = get_current_user_email()
    if guide:
        rows, _ = db_execute(
            "SELECT guide, pathname, anchor, title, scroll_y, scroll_percent, updated_at FROM reading_state_v2 WHERE user_email = ? AND guide = ?",
            (user_email, normalize_guide_name(guide))
        )
    else:
        rows, _ = db_execute(
            "SELECT guide, pathname, anchor, title, scroll_y, scroll_percent, updated_at FROM reading_state_v2 WHERE user_email = ? ORDER BY updated_at DESC LIMIT 1",
            (user_email,)
        )
    
    state = rows[0] if rows else None
    response = jsonify({"ok": True, "state": state})
    response.headers["Cache-Control"] = "no-store"
    return response

@app.route("/api/reading-state", methods=["POST"])
def save_reading_state():
    payload = request.get_json() or {}
    guide = payload.get("guide")
    pathname = payload.get("pathname")
    anchor = payload.get("anchor") or ""
    title = payload.get("title") or ""
    scroll_y = payload.get("scroll_y", 0)
    scroll_percent = payload.get("scroll_percent", 0.0)
    user_email = get_current_user_email()
    
    if not guide or not pathname:
        return "guide and pathname are required", 400
        
    guide = normalize_guide_name(guide)
    now = utc_now()
    db_execute(
        """
        INSERT INTO reading_state_v2 (user_email, guide, pathname, anchor, title, scroll_y, scroll_percent, updated_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(user_email, guide) DO UPDATE SET
            pathname = excluded.pathname,
            anchor = excluded.anchor,
            title = excluded.title,
            scroll_y = excluded.scroll_y,
            scroll_percent = excluded.scroll_percent,
            updated_at = excluded.updated_at
        """,
        (user_email, guide, pathname, anchor, title, int(scroll_y), float(scroll_percent), now)
    )
    return jsonify({"ok": True})

# --- Authentication & Google Sign-In Integrations ---

LOGIN_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Sign In | Zeeshan Hayat</title>
  {% if google_configured %}
  <script src="https://accounts.google.com/gsi/client" async defer></script>
  {% endif %}
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&family=Outfit:wght@600;700;800&display=swap" rel="stylesheet">
  <style>
    :root {
      --bg: #030712;
      --surface: rgba(15, 23, 42, 0.7);
      --surface-border: rgba(255, 255, 255, 0.07);
      --accent: #06b6d4;
      --accent-gradient: linear-gradient(135deg, #06b6d4 0%, #3b82f6 100%);
      --text: #f3f4f6;
      --text-muted: #9ca3af;
      --font-display: 'Outfit', sans-serif;
      --font-body: 'Inter', sans-serif;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: var(--font-body);
      background: var(--bg);
      color: var(--text);
      display: flex;
      justify-content: center;
      align-items: center;
      min-height: 100vh;
      overflow: hidden;
    }
    
    .glow-container {
      position: absolute;
      width: 100vw;
      height: 100vh;
      pointer-events: none;
      z-index: 0;
      overflow: hidden;
    }
    .glow-orb {
      position: absolute;
      width: 600px;
      height: 600px;
      border-radius: 50%;
      background: radial-gradient(circle, rgba(6, 182, 212, 0.08) 0%, rgba(59, 130, 246, 0.03) 50%, transparent 100%);
      filter: blur(100px);
      top: -200px;
      right: -200px;
    }

    .login-container {
      position: relative;
      z-index: 1;
      width: min(420px, 90vw);
      background: var(--surface);
      border: 1px solid var(--surface-border);
      border-radius: 20px;
      padding: 40px;
      box-shadow: 0 20px 40px rgba(0, 0, 0, 0.5);
      backdrop-filter: blur(16px);
      -webkit-backdrop-filter: blur(16px);
    }
    
    .logo {
      font-family: var(--font-display);
      font-size: 24px;
      font-weight: 800;
      text-align: center;
      margin-bottom: 8px;
      background: var(--accent-gradient);
      -webkit-background-clip: text;
      -webkit-text-fill-color: transparent;
    }
    
    .subtitle {
      text-align: center;
      color: var(--text-muted);
      font-size: 14px;
      margin: 0 0 32px;
    }

    .divider {
      display: flex;
      align-items: center;
      text-align: center;
      color: var(--text-muted);
      font-size: 12px;
      margin: 24px 0;
    }
    .divider::before, .divider::after {
      content: '';
      flex: 1;
      border-bottom: 1px solid var(--surface-border);
    }
    .divider:not(:empty)::before { margin-right: 12px; }
    .divider:not(:empty)::after { margin-left: 12px; }

    .form-group {
      margin-bottom: 20px;
    }
    .form-group label {
      display: block;
      margin-bottom: 8px;
      font-size: 13px;
      font-weight: 500;
      color: var(--text-muted);
    }
    .form-group input {
      width: 100%;
      background: rgba(255, 255, 255, 0.02);
      border: 1px solid var(--surface-border);
      border-radius: 8px;
      padding: 12px;
      font-family: inherit;
      font-size: 14px;
      color: #fff;
      transition: all 0.3s;
    }
    .form-group input:focus {
      outline: none;
      border-color: var(--accent);
      background: rgba(255, 255, 255, 0.04);
    }

    .btn-submit {
      width: 100%;
      padding: 14px;
      font-weight: 700;
      border: none;
      background: var(--accent-gradient);
      color: #030712;
      border-radius: 8px;
      cursor: pointer;
      font-family: inherit;
      font-size: 15px;
      transition: all 0.3s;
    }
    .btn-submit:hover {
      transform: translateY(-2px);
      box-shadow: 0 8px 20px rgba(6, 182, 212, 0.2);
    }
    
    /* Center Google Sign In */
    .google-btn-wrapper {
      display: flex;
      justify-content: center;
      margin-bottom: 8px;
    }
    
    .back-link {
      display: block;
      text-align: center;
      margin-top: 24px;
      color: var(--text-muted);
      text-decoration: none;
      font-size: 13px;
      transition: color 0.2s;
    }
    .back-link:hover {
      color: var(--text);
    }
    
    .config-warning {
      background: rgba(239, 68, 68, 0.08);
      border: 1px solid rgba(239, 68, 68, 0.2);
      color: #f87171;
      padding: 12px;
      border-radius: 8px;
      font-size: 12px;
      margin-bottom: 20px;
      line-height: 1.5;
    }
  </style>
</head>
<body>
  <div class="glow-container">
    <div class="glow-orb"></div>
  </div>

  <div class="login-container">
    <div class="logo">Zeeshan Hayat</div>
    <div class="subtitle">Sign in to sync your studying notes & reading progress</div>

    {% if not google_configured %}
    <div class="config-warning">
      <strong>Note:</strong> Google authentication Client ID is not configured on the server. Google Login is disabled. You can still sign in using the Email option below.
    </div>
    {% endif %}

    {% if google_configured %}
    <div class="google-btn-wrapper">
      <div id="g_id_onload"
           data-client_id="{{ google_client_id }}"
           data-context="signin"
           data-ux_mode="redirect"
           data-login_uri="{{ redirect_uri }}"
           data-auto_prompt="false">
      </div>
      <div class="g_id_signin"
           data-type="standard"
           data-shape="rectangular"
           data-theme="filled_dark"
           data-text="signin_with"
           data-size="large"
           data-logo_alignment="left">
      </div>
    </div>
    {% endif %}

    <div class="divider">or continue with email</div>

    <form action="/login/email" method="POST">
      <div class="form-group">
        <label for="email">Email Address</label>
        <input type="email" id="email" name="email" required placeholder="name@example.com">
      </div>
      <button type="submit" class="btn-submit">Sign In with Email</button>
    </form>

    <a href="/" class="back-link">← Back to Homepage</a>
  </div>


</body>
</html>
"""

SESSION_PILL_SCRIPT = """
async function initSessionPill() {
  let user = null;
  try {
    const res = await fetch("/api/auth/status");
    if (res.ok) {
      const data = await res.json();
      if (data.logged_in) {
        user = data.email;
      }
    }
  } catch (err) {
    console.warn("Failed to check auth status:", err);
  }

  const style = document.createElement("style");
  style.textContent = `
    .session-pill-container {
      position: fixed;
      top: 16px;
      right: 16px;
      z-index: 9999;
      font-family: system-ui, -apple-system, sans-serif;
    }
    .session-pill {
      background: rgba(15, 23, 42, 0.85);
      backdrop-filter: blur(12px);
      -webkit-backdrop-filter: blur(12px);
      border: 1px solid rgba(255, 255, 255, 0.08);
      padding: 8px 16px;
      border-radius: 99px;
      display: flex;
      align-items: center;
      gap: 12px;
      box-shadow: 0 4px 20px rgba(0, 0, 0, 0.15);
      font-size: 13px;
      color: #f3f4f6;
    }
    .session-pill a {
      color: #06b6d4;
      text-decoration: none;
      font-weight: 600;
      transition: color 0.2s;
    }
    .session-pill a:hover {
      color: #3b82f6;
    }
    .session-pill .logout-btn {
      color: #ef4444;
    }
    .session-pill .logout-btn:hover {
      color: #f87171;
    }
  `;
  document.head.appendChild(style);

  const container = document.createElement("div");
  container.className = "session-pill-container";

  const pill = document.createElement("div");
  pill.className = "session-pill";

  if (user) {
    pill.innerHTML = `<span>Studying as <strong>${user}</strong></span><a href="/logout" class="logout-btn">Logout</a>`;
  } else {
    pill.innerHTML = `<span>Not signed in</span><a href="/login">Sign In</a>`;
  }

  container.appendChild(pill);
  document.body.appendChild(container);
}
window.addEventListener("DOMContentLoaded", initSessionPill);
"""

def get_google_client_id_from_file() -> str | None:
    config_file = ROOT / "auth_config.json"
    if config_file.exists():
        try:
            with open(config_file, "r") as f:
                data = json.load(f)
                val = data.get("google_client_id")
                return val.strip() if val else None
        except Exception as e:
            print("Failed to read auth_config.json:", e)
    return None

def verify_google_token(id_token: str) -> dict | None:
    try:
        url = f"https://oauth2.googleapis.com/tokeninfo?id_token={urllib.parse.quote(id_token)}"
        req = urllib.request.Request(url, method="GET")
        with urllib.request.urlopen(req, timeout=5) as response:
            data = json.loads(response.read().decode("utf-8"))
            if "error_description" in data:
                print("Google token verification error:", data["error_description"])
                return None
            return data
    except Exception as e:
        print("Failed to verify Google token:", e)
        return None

def get_current_user_email() -> str:
    return session.get("user_email") or "anonymous"

@app.route("/login")
def login_page():
    google_client_id = os.environ.get("GOOGLE_CLIENT_ID") or get_google_client_id_from_file()
    google_configured = bool(google_client_id)
    
    scheme = request.headers.get("X-Forwarded-Proto", "http")
    host = request.headers.get("Host", request.host)
    if "zeehayat.com" in host:
        scheme = "https"
    redirect_uri = f"{scheme}://{host}/api/auth/google/callback"
    
    return render_template_string(
        LOGIN_TEMPLATE,
        google_configured=google_configured,
        google_client_id=google_client_id,
        redirect_uri=redirect_uri
    )

@app.route("/login/email", methods=["POST"])
def login_email():
    email = request.form.get("email", "").strip().lower()
    if not email:
        return "Email is required", 400
    session["user_email"] = email
    return redirect("/")

@app.route("/api/auth/google/callback", methods=["POST"])
def google_callback():
    id_token = request.form.get("credential")
    if not id_token:
        return "Token is missing", 400
        
    user_info = verify_google_token(id_token)
    if not user_info:
        return "Invalid or expired token", 400
        
    email = user_info.get("email")
    if not email:
        return "Email not found in token", 400
        
    session["user_email"] = email.lower()
    session["user_name"] = user_info.get("name", "")
    session["user_picture"] = user_info.get("picture", "")
    return redirect("/")

@app.route("/logout")
def logout():
    session.clear()
    return redirect("/")

@app.route("/api/auth/status")
def auth_status():
    user = session.get("user_email")
    if user:
        return jsonify({"logged_in": True, "email": user})
    return jsonify({"logged_in": False})
        
@app.route("/api/contact", methods=["POST"])
def save_contact_message():
    payload = request.get_json() or {}
    name = str(payload.get("name") or "").strip()[:100]
    email = str(payload.get("email") or "").strip()[:100]
    subject = str(payload.get("subject") or "").strip()[:200]
    message = str(payload.get("message") or "").strip()
    
    if not name or not email or not message:
        return "Name, email, and message are required", 400
        
    now = utc_now()
    db_execute(
        """
        INSERT INTO contact_messages (name, email, subject, message, created_at)
        VALUES (?, ?, ?, ?, ?)
        """,
        (name, email, subject, message, now)
    )
    return jsonify({"ok": True})

@app.route("/<path:filename>")
def serve_static(filename):
    file_path = ROOT / filename
    if filename.endswith(".html"):
        if file_path.exists():
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
            session_script = f"\n<script>\n{SESSION_PILL_SCRIPT}\n</script>\n"
            content = content.replace("</body>", f"{session_script}</body>")
            response = make_response(content)
            response.headers["Cache-Control"] = "no-store"
            return response
            
        stem = file_path.stem
        md_file = find_markdown_file(stem)
        if md_file and md_file.exists():
            html_content = render_markdown_guide(md_file)
            session_script = f"\n<script>\n{SESSION_PILL_SCRIPT}\n</script>\n"
            html_content = html_content.replace("</body>", f"{session_script}</body>")
            response = make_response(html_content)
            response.headers["Cache-Control"] = "no-store"
            return response
            
    if file_path.exists():
        return send_from_directory(ROOT, filename)
        
    return "Not found", 404

@app.route("/robots.txt")
def robots_txt():
    content = """User-agent: *
Allow: /
Allow: /sitemap.xml
Disallow: /api/
Disallow: /logout

Sitemap: https://zeehayat.com/sitemap.xml
"""
    response = make_response(content)
    response.headers["Content-Type"] = "text/plain; charset=utf-8"
    return response

@app.route("/sitemap.xml")
def sitemap_xml():
    base_url = "https://zeehayat.com"
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    
    urls = [
        f"""  <url>
    <loc>{base_url}/</loc>
    <lastmod>{now}</lastmod>
    <changefreq>daily</changefreq>
    <priority>1.0</priority>
  </url>""",
        f"""  <url>
    <loc>{base_url}/regression_guide.html</loc>
    <lastmod>{now}</lastmod>
    <changefreq>weekly</changefreq>
    <priority>0.9</priority>
  </url>"""
    ]
    
    for f in sorted(ROOT.glob("*.md")):
        norm_name = normalize_guide_name(f.name)
        href = f"{f.stem}.html"
        urls.append(f"""  <url>
    <loc>{base_url}/{href}</loc>
    <lastmod>{now}</lastmod>
    <changefreq>weekly</changefreq>
    <priority>0.8</priority>
  </url>""")
        
    xml_content = f"""<?xml version="1.0" encoding="UTF-8"?>
<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
{"\n".join(urls)}
</urlset>
"""
    response = make_response(xml_content)
    response.headers["Content-Type"] = "application/xml; charset=utf-8"
    return response

def main() -> None:
    import sys
    init_db()
    host = "127.0.0.1"
    port = 8765
    if len(sys.argv) > 1:
        try:
            port = int(sys.argv[1])
        except ValueError:
            pass
    print(f"ML Curriculum Notes Hub: http://{host}:{port}/")
    print(f"SQLite database: {DB_FILE}")
    app.run(host=host, port=port)

if __name__ == "__main__":
    main()

