import React, { useState } from 'react'
import './styles.css'

export default function App() {
  const [email, setEmail] = useState('')
  const [message, setMessage] = useState('Please log in')

  // BUG 1 (interaction): onClick is invoked immediately instead of passing a function on small screens
  function submit() {
    if(!email) {
      setMessage('Error: email required')
      return
    }
    setMessage('Logged in!')
  }

  return (
    <div className="container">
      <h1 style={{color:'#777'}}>Welcome</h1> {/* BUG 2 (a11y): low contrast heading */}
      {/* BUG 3 (a11y): label missing htmlFor */}
      <label>Email</label>
      <input id="email" value={email} onChange={(e)=>setEmail(e.target.value)} placeholder="you@example.com" />
      {/* BUG 4 (visual on mobile): pointer-events disabled via CSS */}
      <button id="login" onClick={window.innerWidth < 480 ? submit() : submit} title="Submit form">Login</button>
      {/* BUG 5 (a11y): decorative image without alt text */}
      <img src="https://via.placeholder.com/120x40?text=Logo" />
      <p id="status">{message}</p>
    </div>
  )
}
