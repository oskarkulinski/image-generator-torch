import torch
import matplotlib.pyplot as plt
import datetime
from time import time_ns
import os
import torchvision.utils as vutils
from torcheval.metrics import FrechetInceptionDistance

import parameters as params
from discriminator import Discriminator
from generator import Generator
from parameters import display_amount_height, display_amount_width


class SceneGenerator:
    def __init__(self):
        self.discriminator = Discriminator()
        self.generator = Generator()
        self.generator_optimizer = torch.optim.Adam(self.generator.parameters(),
                                                    1e-4, (0.5, 0.99))
        self.discriminator_optimizer = torch.optim.Adam(self.discriminator.parameters(),
                                                        1e-4, (0.5, 0.99))
        self.loss = torch.nn.BCELoss()
        self.noise = torch.randn(params.display_amount_height * params.display_amount_width, params.noise_dim)
        # noise used for displaying images, set on init to showcase how the generated images change between epochs

    def train(self, train_dataset, epochs):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.generator.to(device)
        self.discriminator.to(device)
        ug = []
        ud = []
        current_datetime = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")
        folder_name = os.path.join("saved_models", current_datetime)
        os.makedirs(folder_name, exist_ok=True)
        for epoch in range(epochs):
            start = time_ns()
            gen_loss_list = []
            disc_loss_list = []
            fid_list = []

            for real_images in train_dataset:
                real_images = real_images.to(device)
                batch_size = real_images.size(0)

                real_labels = torch.ones(batch_size, 1).to(device)
                fake_labels = torch.zeros(batch_size, 1).to(device)

                # ---------------------
                #  Train Discriminator
                # ---------------------
                self.discriminator_optimizer.zero_grad()
                real_output = self.discriminator(real_images)
                real_loss = self.loss(real_output, real_labels)

                noise = torch.randn(batch_size, params.noise_dim).to(device)
                fake_images = self.generator(noise)

                fake_output = self.discriminator(fake_images.detach())
                fake_loss = self.loss(fake_output, fake_labels)

                # Total Discriminator loss
                d_loss = real_loss + fake_loss
                d_loss.backward()
                self.discriminator_optimizer.step()

                # -----------------
                #  Train Generator
                # -----------------

                self.generator_optimizer.zero_grad()

                fake_images = self.generator(noise)

                fake_output = self.discriminator(fake_images)
                g_loss = self.loss(fake_output, real_labels)

                g_loss.backward()
                self.generator_optimizer.step()

                gen_loss_list.append(g_loss)
                disc_loss_list.append(d_loss)
                with torch.no_grad():
                    ug.append([((1e-3 * p.grad).std() / p.data.std()).log10().item() for p in self.generator.parameters()])
                    ud.append([((1e-4 * p.grad).std() / p.data.std()).log10().item() for p in self.discriminator.parameters()])

                if epoch % 10 == 0:
                    fid_list.append(FrechetInceptionDistance(real_images, fake_images, device=device))

            g_loss = sum(gen_loss_list) / len(gen_loss_list)
            d_loss = sum(disc_loss_list) / len(disc_loss_list)
            fid = sum(fid_list) / len(fid_list) if fid_list else 0
            end = time_ns()

            print(f"{epoch}:"
                  f"[D loss: {d_loss:.4f}] [G loss: {g_loss:.4f}] "
                  f"[Time: {(end - start) * 0.000000001:.3f}s]" +
                  f"[FID: {fid:.4f}]")

            if epoch % params.sample_interval == 0:
                self.sample_images(device)
                plt.figure(figsize=(20, 4))
                legends = []
                for i,p in enumerate(self.generator.parameters()):
                    plt.plot([ug[j][i] for j in range(len(ug))])
                    legends.append('G param %d' % i)

                plt.plot([0, len(ud)], [-3, -3], 'k') # these ratios should be ~1e-3, indicate on plot
                plt.legend(legends)
                plt.show()

                plt.figure(figsize=(20, 4))
                legends = []
                for i,p in enumerate(self.discriminator.parameters()):
                    plt.plot([ud[j][i] for j in range(len(ud))])
                    legends.append('D param %d' % i)

                plt.plot([0, len(ud)], [-3, -3], 'k') # these ratios should be ~1e-3, indicate on plot
                plt.legend(legends)
                plt.show()

            if epoch != 0 and epoch % params.save_interval == 0:
                sub_folder_name = os.path.join(folder_name, f"epoch_{epoch}")
                os.makedirs(sub_folder_name, exist_ok=True)
                self.save_models(sub_folder_name, epoch)


    def save_models(self, folder_name, epoch):
        torch.save(self.discriminator.state_dict(), os.path.join(folder_name, "discriminator.pt"))
        torch.save(self.generator.state_dict(), os.path.join(folder_name, "generator.pt"))
        print(f"Models saved for epoch {epoch} at {folder_name}")

    def load_models(self, folder_name):
        discriminator_path = os.path.join(folder_name, "discriminator.h5")
        generator_path = os.path.join(folder_name, "generator.h5")
        self.discriminator.load_state_dict(torch.load(discriminator_path))
        self.generator.load_state_dict(torch.load(generator_path))
        print(f"Models loaded from {folder_name}")

    def sample_images(self, device, num_images=9, figsize=(5,5)):
        """
        Display images generated by the GAN

        Args:
            device: Device to generate images on
            num_images: Number of images to display
            figsize: Size of the figure:
        """
        self.generator.eval()

        with torch.no_grad():
            images = self.generator(self.noise.to(device))
            images = (images + 1) / 2.0

            grid = vutils.make_grid(images[:num_images],
                                    padding=2,
                                    normalize=False,
                                    nrow=display_amount_width)
            grid = grid.cpu().numpy().transpose((1, 2, 0))

            plt.figure(figsize=figsize)
            plt.axis("off")
            plt.imshow(grid)
            plt.show()