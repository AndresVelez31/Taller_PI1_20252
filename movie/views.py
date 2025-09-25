import matplotlib.pyplot as plt
import matplotlib
import io
import urllib, base64
from django.shortcuts import render
from django.http import HttpResponse
from openai import OpenAI
import numpy as np
import os
from dotenv import load_dotenv

from .models import Movie

# Create your views here.

def home(request):
    #return HttpResponse('<h1>Welcome to Home Page</h1>')
    #return render(request, 'home.html')
    #return render(request, 'home.html', {'name': 'Andres Velez'})
    searchTerm = request.GET.get('searchMovie')
    if searchTerm: 
        movies = Movie.objects.filter(title__icontains=searchTerm)  
    else:  
        movies = Movie.objects.all()
    return render(request, 'home.html', {'searchTerm': searchTerm, 'movies': movies, 'name1' : 'Andres Velez'})
    
def about(request):
    #return HttpResponse('<h1>About Us</h1>')
    return render(request, 'about.html')


def statistics_view(request):
    matplotlib.use('Agg')
    years = Movie.objects.values_list('year', flat=True).distinct().order_by('year') # Obtener todos los años de las películas
    movie_counts_by_year = {} # Crear un diccionario para almacenar la cantidad de películas por año
    
    for year in years: # Contar la cantidad de películas por año
        if year:
            movies_in_year = Movie.objects.filter(year=year)
        else:
            movies_in_year = Movie.objects.filter(year__isnull=True)
            year = "None"
        count = movies_in_year.count()
        movie_counts_by_year[year] = count
        
    bar_width = 0.5 # Ancho de las barras
    bar_spacing = 0.5 # Separación entre las barras
    bar_positions = range(len(movie_counts_by_year)) # Posiciones de las barras
    
    # Crear la gráfica de barras
    plt.bar(bar_positions, movie_counts_by_year.values(), width=bar_width, align='center')
    
    # Personalizar la gráfica
    plt.title('Movies per year')
    plt.xlabel('Year')
    plt.ylabel('Number of movies')
    plt.xticks(bar_positions, movie_counts_by_year.keys(), rotation=90)
    
    # Ajustar el espaciado entre las barras
    plt.subplots_adjust(bottom=0.3)
    
    # Guardar la gráfica en un objeto BytesIO
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    plt.close()
    
    # Convertir la gráfica a base64
    image_png = buffer.getvalue()
    buffer.close()
    graphic_year = base64.b64encode(image_png)
    graphic_year = graphic_year.decode('utf-8')
    
    # Peliculas por genero (solo el primer genero por pelicula)
    genre_counts = {}
    for movie in Movie.objects.all():
        if movie.genre:
            first_genre = movie.genre.split(',')[0].strip()
            genre_counts[first_genre] = genre_counts.get(first_genre, 0) + 1
    plt.figure(figsize=(6,4))
    plt.bar(genre_counts.keys(), genre_counts.values(), color='green')
    plt.title('Movies per genre (first only)')
    plt.xlabel('Genre')
    plt.ylabel('Number of Movies')
    plt.xticks(rotation=45)
    plt.tight_layout()
    buffer_genre = io.BytesIO()
    plt.savefig(buffer_genre, format='png')
    buffer_genre.seek(0)
    plt.close()
    image_png_genre = buffer_genre.getvalue()
    buffer_genre.close()
    graphic_genre = base64.b64encode(image_png_genre).decode('utf-8')

    # Renderizar la plantilla stadistics.html con ambas gráficas
    return render(request, 'statistics.html', {'graphic_year': graphic_year, 'graphic_genre': graphic_genre})
    

def signup(request):
    email = request.GET.get('email')
    return render(request, 'signup.html', {'email': email})


def cosine_similarity(a, b):
    """Calcula la similitud de coseno entre dos vectores."""
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def recommendation_view(request):
    """Vista para el sistema de recomendación basado en embeddings."""
    context = {
        'search_prompt': '',
        'recommended_movie': None,
        'similarity_score': None,
        'error_message': None
    }
    
    if request.method == 'POST':
        search_prompt = request.POST.get('search_prompt', '').strip()
        context['search_prompt'] = search_prompt
        
        if search_prompt:
            try:
                # Cargar la API Key de OpenAI
                load_dotenv('openAI.env')
                client = OpenAI(api_key=os.environ.get('openai_apikey'))
                
                # Generar embedding del prompt
                response = client.embeddings.create(
                    input=[search_prompt],
                    model="text-embedding-3-small"
                )
                prompt_emb = np.array(response.data[0].embedding, dtype=np.float32)
                
                # Recorrer la base de datos y comparar similitudes
                best_movie = None
                max_similarity = -1
                
                for movie in Movie.objects.all():
                    # Convertir el embedding almacenado de bytes a numpy array
                    movie_emb = np.frombuffer(movie.emb, dtype=np.float32)
                    similarity = cosine_similarity(prompt_emb, movie_emb)
                    
                    if similarity > max_similarity:
                        max_similarity = similarity
                        best_movie = movie
                
                if best_movie:
                    context['recommended_movie'] = best_movie
                    context['similarity_score'] = round(max_similarity * 100, 2)  # Convertir a porcentaje
                else:
                    context['error_message'] = "No se encontraron películas en la base de datos."
                    
            except Exception as e:
                context['error_message'] = f"Error al procesar la búsqueda: {str(e)}"
        else:
            context['error_message'] = "Por favor ingresa una descripción para buscar películas."
    
    return render(request, 'recommendation.html', context)