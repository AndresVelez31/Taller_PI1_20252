from django.core.management.base import BaseCommand
from movie.models import Movie
import random
import numpy as np

class Command(BaseCommand):
    help = 'Visualiza los embeddings de una película seleccionada al azar.'

    def handle(self, *args, **kwargs):
        movies = list(Movie.objects.all())
        if not movies:
            self.stdout.write(self.style.ERROR('No hay películas en la base de datos.'))
            return
        movie = random.choice(movies)
        self.stdout.write(self.style.SUCCESS(f'Película seleccionada: {movie.title}'))
        # Ejemplo: embedding simple basado en la descripción
        embedding = self.get_embedding(movie.description)
        self.stdout.write(f'Embedding: {embedding}')

    def get_embedding(self, text):
        arr = np.array([ord(c) for c in text])
        if len(arr) == 0:
            return np.zeros(10)
        emb = arr[:10] if len(arr) >= 10 else np.pad(arr, (0, 10-len(arr)), 'constant')
        return emb / np.linalg.norm(emb)
