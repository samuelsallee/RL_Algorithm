from header import *
import bullet
from enemy import enemy
import draw
import player


class Game_Class:
    def __init__(self, show):
        self.testing = 0
        self.screen_WIDTH: int = 800
        self.screen_HEIGHT: int = 600
        self.background_WIDTH = 500
        self.background_HEIGHT = 500
        self.reward = 0
        self.enemies_killed = 0
        self.draw_and_show = show

        self.action = 0

        self.go_button_Width: int = 200
        self.go_button_Height: int = 45

        self.FPS: int = 60

        self.money = 0

        self.enemyList = list()

        if self.draw_and_show:
            if pygame.init() == 0:
                print("PyGame could not initialize")

            self.logo = pygame.image.load("8bitlink.png")
            self.pause = pygame.transform.scale(pygame.image.load("Pause.png"), (600, 150))
            pygame.display.set_icon(self.logo)
            pygame.display.set_caption("# Learn to Code")

            self.Gunfire = pygame.mixer.Sound('Silent.wav')
            self.screen = pygame.display.set_mode((self.screen_WIDTH, self.screen_HEIGHT), RESIZABLE)
            self.FramesPerSecond = pygame.time.Clock()
            self.font = pygame.font.SysFont('comicsans', 30, True, True)  # Initializes self.font

            self.background = pygame.transform.scale(pygame.image.load("Stone Walls\stone_wall_1_small.png"),
                                                     (self.background_HEIGHT, self.background_WIDTH))
            self.playerImage = pygame.transform.scale(pygame.image.load("survivor-idle_rifle_0.png"), (57, 40))
            self.player_one = player.Player(sprite=self.playerImage, SCREEN_WIDTH=self.screen_WIDTH, SCREEN_HEIGHT=self.screen_HEIGHT)
        else:
            self.player_one = player.Player(SCREEN_WIDTH=self.screen_WIDTH, SCREEN_HEIGHT=self.screen_HEIGHT)


        self.Goblin = enemy(random.random(), 0, 64, 64, 2, 550, 100)


        self.game_quit: bool = False
        self.mainMenu: bool = True

        self.buttonWidth: int = 250
        self.buttonHeight: int = 65
        self.score = 0
        self.start = float(round(time.time()))
        self.running: bool = True
        self.background_x: int = -self.background_WIDTH
        self.background_y: int = -self.background_HEIGHT
        self.xDelta: float = 0
        self.yDelta: float = 0
        self.extra_enemies: int = 0
        self.wave: int = -1
        self.bullet_damage: int = 100
        self.bullet_speed: int = 10
        self.number_of_frames_shown: int = 0
        self.rotation = 0

    def hit_logic(self, person1):
        for enemy_object in self.enemyList:
            for bullet_object in bullet.bulletList:
                if enemy_object.box_x[0] < bullet_object.locationx < enemy_object.box_x[1]:
                    if enemy_object.box_y[0] < bullet_object.locationy < enemy_object.box_y[1]:
                        bullet_object.locationx = -6000
                        enemy_object.health -= bullet_object.damage
                        if enemy_object.health < 1:
                            self.money += enemy_object.health_total / 10
                            try:
                                self.enemyList.remove(enemy_object)
                                self.enemies_killed += 1
                            except:
                                pass
                            self.score = self.score + 5  # increases self.score by 5 for every kill

            player_to_enemy_distance_tuple = (
                person1.player_center[0] - enemy_object.center[0], person1.player_center[1] - enemy_object.center[1])
            p_to_e_distance = player_to_enemy_distance_tuple[0] * player_to_enemy_distance_tuple[0] + \
                              player_to_enemy_distance_tuple[1] * player_to_enemy_distance_tuple[1]
            p_to_e_distance = math.sqrt(p_to_e_distance)
            if self.testing == 0:
                if p_to_e_distance < 25:
                    return False
        return True

    def set_background(self):
        if self.draw_and_show:
            rand_num: int = int(random.random() * 10)
            if rand_num < 5:
                self.background = pygame.image.load("backgrounddetailed1_flower.png")
            else:
                self.background = pygame.image.load("backgrounddetailed1.png")

    def step(self):
        self.number_of_frames_shown += 1
        if len(self.enemyList) == 0:
            self.wave += 1
            multiplier = 1
            if self.wave < 60:
                multiplier = self.wave * 10
            i: int = 0
            self.now = float(round(time.time()))
            self.health = 100 + (self.now - self.start) / 2
            while i < self.extra_enemies:
                around = float(random.random() * 10)
                self.gob_x = math.cos(around) * (multiplier * random.random() * (
                        1 + (self.now - self.start) / 10) + 700) + self.screen_WIDTH / 2
                self.gob_y = math.sin(around) * (multiplier * random.random() * (
                        1 + (self.now - self.start) / 10) + 700) + self.screen_HEIGHT / 2
                self.Goblin = enemy(self.gob_x, self.gob_y, 64, 64, 2, 550, self.health)
                self.enemyList.append(self.Goblin)
                i += 1
            self.extra_enemies += 1
        if self.draw_and_show:
            draw.draw(self.rotation, self.screen_WIDTH, self.screen_HEIGHT, self.screen, self.enemyList,
                      self.background, self.xDelta, self.yDelta,
                      self.background_x, self.background_y, self.background_WIDTH, self.background_HEIGHT,
                      self.player_one)
        else:
            draw.do_not_draw(self.xDelta, self.yDelta, self.enemyList, self.player_one)

        if self.action == 0:
            self.xDelta = 0
            self.yDelta = 0
        elif self.action == 1:
            self.xDelta = -5
            self.yDelta = 0
        elif self.action == 2:
            self.xDelta = 5
            self.yDelta = 0
        elif self.action == 3:
            self.yDelta = -5
            self.xDelta = 0
        elif self.action == 4:
            self.yDelta = 5
        elif self.action == 5:
            self.rotation -= 12
            self.xDelta = 0
            self.yDelta = 0
        elif self.action == 6:
            self.rotation += 12
            self.xDelta = 0
            self.yDelta = 0
        elif self.action == 7:
            self.xDelta = 0
            self.yDelta = 0
            self.direct = self.rotation % 360 - 180
            self.buldeltay = math.sin(math.radians(-self.direct)) * 10
            self.buldeltax = math.cos(math.radians(-self.direct)) * 10
            new_bullet = bullet.Bullet(self.bullet_damage,
                                       self.direct,
                                       self.bullet_speed,
                                       self.player_one.position_x + (math.cos(math.radians(-self.direct) + .45) * 21),
                                       self.player_one.position_y + (math.sin(math.radians(-self.direct) + .45) * 21),
                                       self.buldeltax,
                                       self.buldeltay)
            bullet.bulletList.append(new_bullet)

        #########################################################
        #         Handles idle and walking animation
        if self.xDelta == 0 and self.yDelta == 0:
            self.player_one.walking_counter = 0
            if self.number_of_frames_shown % 3 == 0:
                self.player_one.idle_counter += 1
                self.player_one.sprite_to_show_while_idle()

        else:
            self.player_one.idle_counter = 0
            if self.number_of_frames_shown % 3 == 0:
                self.player_one.walking_counter += 1
                self.player_one.sprite_to_show_while_walking()

        self.background_x -= self.xDelta
        self.background_y -= self.yDelta
        self.player_one.overall_position_x += self.xDelta
        self.player_one.overall_position_y -= self.yDelta
        if self.background_x >= 0:
            self.background_x = -self.background_WIDTH
        if self.background_y >= 0:
            self.background_y = -self.background_HEIGHT

        self.running = self.hit_logic(self.player_one)

        if self.draw_and_show:
            draw.draw_useful_information(self.screen, self.font, self.score, self.wave, self.money, self.player_one)

            pygame.display.update()
            self.FramesPerSecond.tick(self.FPS)

        if self.running == False and self.draw_and_show == True:
            pygame.quit()
