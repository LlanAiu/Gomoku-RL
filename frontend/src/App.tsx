// builtin

// external
import { Link } from 'react-router'

// internal
import './App.css'


export default function App() {
    return (
        <div className="home-root">
            <div className="home-card">
                <h1 className="home-title">Gomoku Time!</h1>
                <p className="home-desc">A word of warning though, do try to keep your expectations low y'all...</p>
                <Link to="/game" className="home-cta">Bet -- let's go!</Link>
            </div>
        </div>
    );
}
