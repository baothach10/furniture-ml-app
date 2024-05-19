export default function MainPage() {
  fetch('http://localhost:8001/api/v1/')

  return (
    <div>
      <img src="http://localhost:8001/static/image/testing.jpg"></img>
      <h1>
        Hello
      </h1>
    </div>
  )
}