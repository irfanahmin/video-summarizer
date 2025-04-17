// script.js

const form = document.getElementById('upload-form');
const videoInput = document.getElementById('video-input');
const progress = document.getElementById('progress');
const results = document.getElementById('results');
const transcriptEl = document.getElementById('transcript');
const summaryAbsEl = document.getElementById('summary-abs');
const summaryExtEl = document.getElementById('summary-ext');

form.addEventListener('submit', async e => {
  e.preventDefault();
  const file = videoInput.files[0];
  if (!file) return;

  progress.classList.remove('hidden');
  results.classList.add('hidden');

  const formData = new FormData();
  formData.append('file', file);

  try {
    const resp = await fetch('/process', {
      method: 'POST',
      body: formData
    });
    if (!resp.ok) throw new Error(`Server returned ${resp.status}`);
    const data = await resp.json();

    transcriptEl.textContent = data.transcript;
    summaryAbsEl.textContent = data.summary_abstractive;
    summaryExtEl.textContent = data.summary_extractive;

    results.classList.remove('hidden');
  } catch (err) {
    alert('Error: ' + err.message);
  } finally {
    progress.classList.add('hidden');
  }
});
