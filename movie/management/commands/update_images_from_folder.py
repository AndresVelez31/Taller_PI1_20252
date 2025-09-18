from django.core.management.base import BaseCommand
from movie.models import Movie
import os

class Command(BaseCommand):
    help = 'Actualiza el campo image de las películas usando los archivos en media/movie/images/'

    def handle(self, *args, **kwargs):
        images_folder = 'media/movie/images/'
        images = os.listdir(images_folder)
        updated = 0
        for movie in Movie.objects.all():
            # Buscar imagen cuyo nombre contenga el título (ignorando mayúsculas/minúsculas y espacios)
            movie_title_clean = movie.title.lower().replace(' ', '')
            found_image = None
            for img in images:
                img_clean = img.lower().replace(' ', '')
                if movie_title_clean in img_clean:
                    found_image = img
                    break
            if found_image:
                movie.image = f'movie/images/{found_image}'
                movie.save()
                updated += 1
        self.stdout.write(self.style.SUCCESS(f'Imágenes actualizadas para {updated} películas.'))