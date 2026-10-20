/** @type {import('next').NextConfig} */
module.exports = {
  reactStrictMode: true,
  env: {
    API_BASE: process.env.API_BASE || "http://127.0.0.1:8000",
  },
};
