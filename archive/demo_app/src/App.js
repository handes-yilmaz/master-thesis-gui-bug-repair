import React, { useState } from 'react';
import './App.css';

function App() {
  const [count, setCount] = useState(0);

  return (
    <div className="App">
      <h1>GUI Bug Demo</h1>
      <p>Click the button to increment count:</p>
      <div style={{ position: 'relative', display: 'inline-block' }}>
        <button id="increment-btn" onClick={() => setCount(count + 1)}>
          Increment
        </button>
        {/* BUG: Invisible overlay blocking clicks */}
        <div style={{
          position: 'absolute',
          top: 0,
          left: 0,
          width: '100%',
          height: '100%',
          backgroundColor: 'transparent'
        }} title="overlay"></div>
      </div>
      <p>Count: {count}</p>
    </div>
  );
}

export default App;
