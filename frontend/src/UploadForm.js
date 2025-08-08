import React, { useState } from 'react';
import axios from 'axios';

let apiBase = process.env.VITE_API_BASE || 'http://localhost:8000';

// Access import.meta.env only if supported — use eval so Jest's parser doesn't choke
try {
  const viteEnv = eval('import.meta.env'); // bypass parse-time syntax check
  if (viteEnv && viteEnv.VITE_API_BASE) {
    apiBase = viteEnv.VITE_API_BASE;
  }
} catch (_) {
  // ignore if not running in a Vite environment
}

const UploadForm = () => {
  const [file, setFile] = useState(null);
  const [results, setResults] = useState(null);

  const handleFileChange = (e) => {
    setFile(e.target.files[0]);
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!file) return;

    const formData = new FormData();
    formData.append('file', file);

    try {
      const response = await axios.post(`${apiBase}/upload`, formData, {
        headers: { 'Content-Type': 'multipart/form-data' }
      });
      setResults(response.data);
    } catch (error) {
      console.error(error);
      setResults({ error: 'Upload failed' });
    }
  };

  return (
    <div>
      <form onSubmit={handleSubmit}>
        <label htmlFor="csvFile">Upload CSV File:</label>
        <input
          id="csvFile"
          type="file"
          accept=".csv"
          onChange={handleFileChange}
        />
        <button type="submit">Upload CSV</button>
      </form>
      {results && (
        <div>
          <h2>Results:</h2>
          <pre>{JSON.stringify(results, null, 2)}</pre>
        </div>
      )}
    </div>
  );
};

export default UploadForm;
