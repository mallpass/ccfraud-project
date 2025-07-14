import { render, screen } from '@testing-library/react';
import App from './App';

test('renders the app title and upload form', () => {
  render(<App />);
  
  // Check for main title
  const heading = screen.getByText(/credit card fraud detector/i);
  expect(heading).toBeInTheDocument();

  // Check for file input label
  const label = screen.getByLabelText(/upload csv file/i);
  expect(label).toBeInTheDocument();

  // Check specifically for the upload button
  const uploadButton = screen.getByRole('button', { name: /upload csv/i });
  expect(uploadButton).toBeInTheDocument();
});
