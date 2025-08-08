import React from 'react';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import axios from 'axios';
import UploadForm from './UploadForm';

jest.mock('axios');

const expectedBase = process.env.VITE_API_BASE || 'http://localhost:8000';

test('renders upload form', () => {
  render(<UploadForm />);
  expect(screen.getByText('Upload CSV')).toBeInTheDocument();
  expect(screen.getByLabelText(/upload csv file/i)).toBeInTheDocument();
});

test('handles file upload submit', async () => {
  const mockResponse = { data: { message: 'Success' } };
  axios.post.mockResolvedValue(mockResponse);

  render(<UploadForm />);
  const fileInput = screen.getByLabelText(/upload csv file/i);
  const file = new File(['dummy content'], 'test.csv', { type: 'text/csv' });

  fireEvent.change(fileInput, { target: { files: [file] } });

  const button = screen.getByText('Upload CSV');
  fireEvent.click(button);

  await waitFor(() => {
    expect(axios.post).toHaveBeenCalledWith(
      `${expectedBase}/upload`,
      expect.any(FormData),
      expect.objectContaining({
        headers: { 'Content-Type': 'multipart/form-data' }
      })
    );

    expect(screen.getByText(/results/i)).toBeInTheDocument();
    expect(screen.getByText(/success/i)).toBeInTheDocument();
  });
});
